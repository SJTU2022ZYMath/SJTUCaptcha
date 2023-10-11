import os
from os import listdir, path
from shutil import copyfile

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import DataLoader, Dataset, random_split


def _weights_init(m):
    if isinstance(m, torch.nn.Linear) or isinstance(m, torch.nn.Conv2d):
        torch.nn.init.kaiming_normal_(m.weight)


class LambdaLayer(torch.nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)


class BasicBlock(torch.nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, option="A"):
        super(BasicBlock, self).__init__()
        self.conv1 = torch.nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = torch.nn.BatchNorm2d(planes)
        self.conv2 = torch.nn.Conv2d(
            planes, planes, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = torch.nn.BatchNorm2d(planes)

        self.shortcut = torch.nn.Sequential()
        if stride != 1 or in_planes != planes:
            if option == "A":
                """
                For CIFAR10 ResNet paper uses option A.
                """
                self.shortcut = LambdaLayer(
                    lambda x: torch.nn.functional.pad(
                        x[:, :, ::2, ::2],
                        (0, 0, 0, 0, planes // 4, planes // 4),
                        "constant",
                        0,
                    )
                )
            elif option == "B":
                self.shortcut = torch.nn.Sequential(
                    torch.nn.Conv2d(
                        in_planes,
                        self.expansion * planes,
                        kernel_size=1,
                        stride=stride,
                        bias=False,
                    ),
                    torch.nn.BatchNorm2d(self.expansion * planes),
                )

    def forward(self, x):
        out = torch.nn.functional.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = torch.nn.functional.relu(out)
        return out


class ResNet(torch.nn.Module):
    def __init__(self, block, num_blocks):
        super(ResNet, self).__init__()
        self.in_planes = 16

        self.conv1 = torch.nn.Conv2d(
            1, self.in_planes, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn1 = torch.nn.BatchNorm2d(self.in_planes)
        self.layer1 = self._make_layer(block, self.in_planes, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(
            block, self.in_planes * 2, num_blocks[1], stride=2
        )
        self.layer3 = self._make_layer(
            block, self.in_planes * 2, num_blocks[2], stride=2
        )
        self.linear1 = torch.nn.Linear(self.in_planes, 26)
        self.linear2 = torch.nn.Linear(self.in_planes, 26)
        self.linear3 = torch.nn.Linear(self.in_planes, 26)
        self.linear4 = torch.nn.Linear(self.in_planes, 26)
        self.linear5 = torch.nn.Linear(self.in_planes, 27)

        self.apply(_weights_init)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, option="B"))
            self.in_planes = planes * block.expansion

        return torch.nn.Sequential(*layers)

    def forward(self, x):
        out = torch.nn.functional.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        # out = torch.nn.functional.avg_pool2d(out, out.size()[2:])
        out = torch.nn.functional.avg_pool2d(out, (10, 25), stride=(10, 25))
        out = out.view(out.size(0), -1)
        out1 = self.linear1(out)
        out2 = self.linear2(out)
        out3 = self.linear3(out)
        out4 = self.linear4(out)
        out5 = self.linear5(out)

        return out1, out2, out3, out4, out5


def resnet20():
    return ResNet(BasicBlock, [3, 3, 3])


# 以下是训练代码


# Train config
USE_CUDA = False  # 是否使用英伟达®CUDA架构 GPU训练
USE_MPS = False  # 是否使用 Apple Silicon 芯片
TEST_FACTOR = 0.2
# Hyper parameters
start_epoch = 0  # 起始轮数
end_epoch = 75  # 结束轮数
lr = 0.01
batch_size = 640  # 每次训练同时使用的数据条数。根据显卡内存调整这个值。记得相应调整lr。


# Also check CHECKPOINT SETTINGS below.


class CaptchaSet(Dataset):
    def __init__(self, root, transform):
        self._table = [0] * 156 + [1] * 100
        self.transform = transform

        self.root = root
        self.imgs = listdir(root)

    @staticmethod
    def _get_label_from_fn(fn):
        raw_label = fn.split("_")[0]
        labels = [ord(char) - ord("a") for char in raw_label]
        if len(labels) == 4:
            labels.append(26)
        return labels

    def __getitem__(self, idx):
        img = Image.open(path.join(self.root, self.imgs[idx])).convert("L")
        img = img.point(self._table, "1")

        label = CaptchaSet._get_label_from_fn(self.imgs[idx])

        if self.transform:
            img = self.transform(img)

        return img, label

    def __len__(self):
        return len(self.imgs)


mps_device = None
if USE_MPS:
    mps_device = torch.device("mps")


def transfer_to_device(x):
    if USE_CUDA:
        return x.cuda()
    elif mps_device is not None:
        return x.to(mps_device)
    else:
        return x


# Data pre-processing
print("==> Preparing data..")
transform = transforms.ToTensor()

dataset = CaptchaSet(root="labelled", transform=transform)
test_count = int(len(dataset) * TEST_FACTOR)
train_count = len(dataset) - test_count
train_set, test_set = random_split(dataset, [train_count, test_count])
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=True)

# Model
print("==> Building model..")
model = resnet20()
model = transfer_to_device(model)

optimizer = optim.Adam(model.parameters(), lr=lr)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3)
criterion = nn.CrossEntropyLoss()
criterion = transfer_to_device(criterion)

# CHECKPOINT SETTINGS
# If you want to restore training (instead of training from beginning),
# you can continue training based on previously-saved models
# WARNING: BEWARE that there may be some problems with this implementation currently. You may get inconsistent results.
restore_model_path = None
# restore_model_path = 'checkpoint/ckpt_0_acc_0.000000.pth'
if restore_model_path:
    checkpoint = torch.load(restore_model_path)
    model.load_state_dict(checkpoint["net"])
    start_epoch = checkpoint["epoch"] + 1
    optimizer.load_state_dict(checkpoint["optimizer"])
    scheduler.load_state_dict(checkpoint["scheduler"])


def tensor_to_captcha(tensors):
    rtn = ""
    for tensor in tensors:
        if int(tensor) == 26:
            rtn += " "
        else:
            rtn += chr(ord("a") + int(tensor))

    return rtn


def train(epoch):
    model.train()
    train_loss = 0
    correct = 0
    total = 0
    per_char_correct = 0
    per_char_total = 0
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs = transfer_to_device(inputs)

        # Learn and predict
        optimizer.zero_grad()
        outputs = model(inputs)

        targets = [transfer_to_device(target) for target in targets]

        # Calculate loss
        losses = []
        for idx in range(len(outputs)):
            losses.append(criterion(outputs[idx], targets[idx]))
        loss = sum(losses)
        loss.backward()
        train_loss += loss.item()

        # Optimizer step
        optimizer.step()

        # Calculate accuracy (sentence based and char based)
        predicted = torch.stack([tensor.max(1)[1] for tensor in outputs], 1)
        targets_stacked = torch.stack(targets, 1)
        per_char_total += targets[0].size(0) * 5
        per_char_correct += predicted.eq(targets_stacked).sum().item()
        total += targets[0].size(0)
        correct += torch.all(predicted.eq(targets_stacked), 1).sum().item()
        batch_idx_last = batch_idx

        # Report statistics
        print(
            "Epoch [%d] Batch [%d/%d] Loss: %.3f | Traininig Acc: [Sentence] %.3f%% (%d/%d) [Char] %.3f%% (%d/%d)"
            % (
                epoch,
                batch_idx + 1,
                len(train_loader),
                train_loss / (batch_idx + 1),
                100.0 * correct / total,
                correct,
                total,
                100.0 * per_char_correct / per_char_total,
                per_char_correct,
                per_char_total,
            )
        )

    # Return train_loss for lr_scheduler
    return train_loss / (batch_idx_last + 1)


def test(epoch):
    print("==> Testing...")
    model.eval()
    total = 0
    correct = 0
    per_char_correct = 0
    per_char_total = 0
    with torch.no_grad():
        ##### TODO: calc the test accuracy #####
        # Hint: You do not have to update model parameters.
        #       Just get the outputs and count the correct predictions.
        #       You can turn to `train` function for help.
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs = transfer_to_device(inputs)
            targets = [transfer_to_device(target) for target in targets]

            # Predict
            outputs = model(inputs)

            # Calculate accuracy (sentence-based and char-based)
            predicted = torch.stack([tensor.max(1)[1] for tensor in outputs], 1)
            targets_stacked = torch.stack(targets, 1)
            total += targets[0].size(0)
            correct += torch.all(predicted.eq(targets_stacked), 1).sum().item()
            per_char_total += targets[0].size(0) * 5
            per_char_correct += predicted.eq(targets_stacked).sum().item()
        acc = 100.0 * correct / total
        per_char_acc = 100.0 * per_char_correct / per_char_total
        ########################################
    # Save checkpoint.
    print("Test Acc: [Sentence] %f [Char] %f" % (acc, per_char_acc))
    print("Saving..")
    state = {
        "net": model.state_dict(),
        "scheduler": scheduler.state_dict(),
        "optimizer": optimizer.state_dict(),
        "acc": acc,
        "epoch": epoch,
    }
    if not os.path.isdir("checkpoint"):
        os.mkdir("checkpoint")
    torch.save(state, "./checkpoint/ckpt_%d_acc_%f.pth" % (epoch, acc))

    return "./checkpoint/ckpt_%d_acc_%f.pth" % (epoch, acc)


for epoch in range(start_epoch, end_epoch + 1):
    train_loss = train(epoch)
    ckpt_file = test(epoch)
    print(f"train_loss: {train_loss}")

    # Scheduler step
    scheduler.step(train_loss)

copyfile(ckpt_file, "./ckpt.pth")
