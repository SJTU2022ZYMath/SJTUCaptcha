import os  # 操作系统接口
import shutil  # shutil.copyfile 只用了这个，复制功能

import torch
import torchvision.transforms
import PIL.Image  # 图片读取


def _weights_init(m):  # 这个函数 _weights_init 用于使用 Kaiming 正态化来标准化线性层和卷积层的权重。
    if isinstance(m, torch.nn.Linear) or isinstance(m, torch.nn.Conv2d):
        torch.nn.init.kaiming_normal_(m.weight)


class LambdaLayer(
    torch.nn.Module
):  # LambdaLayer 是一个 PyTorch 模块，表示一个 Lambda 层。在这个实现中，它用于残差块中的填充。
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)


class BasicBlock(
    torch.nn.Module
):  # BasicBlock 是 ResNet 模型的基本构建块。它包括两个卷积层和一个快速连接层。连接层有助于训练更深的网络。
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, option="A"):
        "输入一个模型的 in_planes, planes, stride=1, option='A'"
        super(BasicBlock, self).__init__()
        self.conv1 = torch.nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        # 第一个卷积层，　planes好像是输入的大小？不知道（悲）
        # in_planes 是进入的通道数 ，planes是输出通道数
        # kernel_size是卷积核大小
        # stride 步幅控制 互相关函数 的步幅 ，啥意思？啥意思？啥意思？啥意思？啥意思？啥意思？啥意思？
        # padding 是在 输入矩阵 边缘 的填充宽度
        # bias：如果为真，将可学习的偏差添加到 输出。
        self.bn1 = torch.nn.BatchNorm2d(planes)
        # https://pytorch.org/docs/stable/generated/torch.nn.BatchNorm2d.html#torch.nn.BatchNorm2d
        # 自己看吧，机翻：对 4D 输入（2D 输入的小批量）应用批量归一化 具有额外的通道维度）
        # 其实就是归一化，归一化就是把数据转化到[-1,1]或[0,1]区间
        self.conv2 = torch.nn.Conv2d(
            planes, planes, kernel_size=3, stride=1, padding=1, bias=False
        )  # 第二个卷积层，接归一化。planes好像是输入的大小？不知道（悲）TODO
        self.bn2 = torch.nn.BatchNorm2d(planes)

        # shortcut 是补回来残差神经网络减掉的输入值
        self.shortcut = torch.nn.Sequential()  # 这TM是啥？
        if stride != 1 or in_planes != planes:  # 如果stride不是默认（1）或者in_planes != planes
            #
            if option == "A":
                """
                For CIFAR10 ResNet paper uses option A.
                """
                self.shortcut = LambdaLayer(
                    lambda x: torch.nn.functional.pad(
                        x[:, :, ::2, ::2],  # x的前两个全部取，后两个 隔一个取一个
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

    def forward(self, x):  # 正向计算
        """
        x => conv1 -> bn1 -> relu -> conv2 -> bn2
                                               +  ------>relu ==> out
        x --------------------------------> shortcut
        """
        out = torch.nn.functional.relu(self.bn1(self.conv1(x)))  # relu函数，终于能看懂一个了
        out = self.bn2(self.conv2(out))  #
        out += self.shortcut(x)
        out = torch.nn.functional.relu(out)
        return out


class ResNet(torch.nn.Module):  # ResNet 是主要的残差网络（ResNet）类，它使用 BasicBlock 构建块定义整体架构
    def __init__(self, block, num_blocks):
        super(ResNet, self).__init__()  # 使用父类nn.Module的初始化
        self.in_planes = 16  # 输入通道数

        self.conv1 = torch.nn.Conv2d(
            1, self.in_planes, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn1 = torch.nn.BatchNorm2d(self.in_planes)

        self.layer1 = self._make_layer(block, self.in_planes, num_blocks[0], stride=1)
        # ResNet的第一块：通道16->16

        self.layer2 = self._make_layer(
            block, self.in_planes * 2, num_blocks[1], stride=2
        )
        # ResNet的第二块：通道16->32
        self.layer3 = self._make_layer(
            block, self.in_planes * 2, num_blocks[2], stride=2
        )
        # ResNet的第三块：通道16->32
        self.linear1 = torch.nn.Linear(self.in_planes, 26)  # 输出通道数in_planes，输出26个通道
        self.linear2 = torch.nn.Linear(self.in_planes, 26)
        self.linear3 = torch.nn.Linear(self.in_planes, 26)
        self.linear4 = torch.nn.Linear(self.in_planes, 26)
        self.linear5 = torch.nn.Linear(self.in_planes, 27)
        # 输出通道数in_planes，输出27个通道(a~z+' ')

        self.apply(_weights_init)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        # 开始的步幅按设定，后面全是1
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, option="B"))
            self.in_planes = planes * block.expansion  # 基本上是*1

        return torch.nn.Sequential(*layers)  # 这一对层用sequntial处理后返回

    def forward(self, x):  # 正向计算
        out = torch.nn.functional.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        # ResNet 3块
        # out = torch.nn.functional.avg_pool2d(out, out.size()[2:])
        out = torch.nn.functional.avg_pool2d(out, (10, 25), stride=(10, 25))

        # 平均
        out = out.view(out.size(0), -1)
        out1 = self.linear1(out)
        out2 = self.linear2(out)
        out3 = self.linear3(out)
        out4 = self.linear4(out)
        out5 = self.linear5(out)

        return out1, out2, out3, out4, out5  # 输出5个分类值


def resnet20():
    return ResNet(
        BasicBlock, [3, 3, 3]
    )  # resnet20 是一个辅助函数，用于创建一个带有指定数量块的 ResNet-20 模型。


# 以下是训练代码
# from torch.utils.data import Dataset, random_split, DataLoader


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


class CaptchaSet(
    torch.utils.data.Dataset
):  # CaptchaSet 是一个自定义 PyTorch 数据集类，用于加载和预处理验证码数据集。储存了一系列图片的文件名。
    def __init__(self, root, transform):
        "root: 数据集所在路径。Transform用于处理图片"
        self._table = [0] * 156 + [1] * 100  # [156个0,100个1]
        self.transform = transform

        self.root = root  # 图片所在文件夹
        self.imgs = os.listdir(root)  # 图片文件路径列表(List)

    @staticmethod  # 静态方法，不可以通过实例访问，直接使用<类名>.<方法>()调用
    def _get_label_from_fn(fn):
        '将文件名"ansdl_1290.png"转化为分类结果"ansdl"'
        raw_label = fn.split("_")[0]
        labels = [ord(char) - ord("a") for char in raw_label]
        if len(labels) == 4:
            labels.append(26)  # 0~25代表a~z，26在末尾作为4为验证码的占位符
        return labels

    def __getitem__(self, idx):
        "按索引，获取图片和分类"
        img = PIL.Image.open(os.path.join(self.root, self.imgs[idx])).convert(
            "L"
        )  # 读取图片，转化为 灰度值图片
        img = img.point(
            self._table, "1"
        )  # 注意self._table的值[156个0,100个1]，相当于：0~155->0,156~255->1
        # PIL.Image.point(self,list)相当于：对于每个像素点，作用：x=list[x]，如果是RBG则是……别的方法
        # 如果是PIL.Image.point(self,func)函数则是x=func(x)

        label = CaptchaSet._get_label_from_fn(self.imgs[idx])  # 获取分类结果

        if self.transform:
            img = self.transform(img)  # 不知道这个transform是干什么的

        return img, label  # 返回(图片,分类)对

    def __len__(self):
        "获取长度"
        return len(self.imgs)


if USE_MPS:
    mps_device = torch.device("mps")
else:
    mps_device = None


def transfer_to_device(x):  # 切换设备……核显别看了
    if USE_CUDA:
        return x.cuda()
    elif mps_device is not None:
        return x.to(mps_device)
    else:
        return x


# 预处理图像
print("==> Preparing data..")
transform = torchvision.transforms.ToTensor()  # 定义了诸如将图像转换为张量等转换。数据集被分成训练集和测试集。

dataset = CaptchaSet(root="labelled", transform=transform)
test_count = int(len(dataset) * TEST_FACTOR)
train_count = len(dataset) - test_count
train_set, test_set = torch.utils.data.random_split(dataset, [train_count, test_count])
train_loader = torch.utils.data.DataLoader(
    train_set, batch_size=batch_size, shuffle=True
)  # 创建了数据加载器，以指定的批量大小遍历训练和测试集
test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=True)

# Model
print("==> Building model..")
model = resnet20()  # 使用 resnet20 函数初始化 ResNet 模型。设置了优化器（Adam）和学习率调度器
model = transfer_to_device(model)

optimizer = torch.optim.Adam(model.parameters(), lr=lr)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3)
criterion = torch.nn.CrossEntropyLoss()
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
        ##### TODO: calc the test accuracy #####（TODO(画大饼的人):所画的大饼 是一种通用的python注释）
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
    torch.save(
        state, "./checkpoint/ckpt_%d_acc_%f.pth" % (epoch, acc)
    )  # 保存模型检查点，包括模型状态、优化器状态、epoch 和准确率。

    return "./checkpoint/ckpt_%d_acc_%f.pth" % (epoch, acc)


for epoch in range(start_epoch, end_epoch + 1):
    train_loss = train(epoch)
    ckpt_file = test(epoch)
    print(f"train_loss: {train_loss}")

    # Scheduler step
    scheduler.step(train_loss)

shutil.copyfile(ckpt_file, "./ckpt.pth")
