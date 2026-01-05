"""
@Author : Keep_Trying_Go
@Major  : Computer Science and Technology
@Hobby  : Computer Vision
@Time   : 2026/1/5-13:44
@CSDN   : https://blog.csdn.net/Keep_Trying_Go?spm=1010.2135.3001.5421
"""

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from accelerate import Accelerator
import os

# ----------------------------
# 1. 配置 & 初始化 Accelerator
# ----------------------------
def main():
    # 可通过 accelerate config 生成配置文件，或直接在代码中指定
    accelerator = Accelerator(
        mixed_precision="fp16",           # 启用混合精度（若 GPU 支持）
        gradient_accumulation_steps=2,    # 梯度累积步数
        log_with="all",                   # 自动支持 TensorBoard/W&B（如有安装）
        project_dir="./cifar100_output"
    )

    # 仅主进程创建输出目录
    if accelerator.is_main_process:
        os.makedirs(accelerator.project_dir, exist_ok=True)

    # ----------------------------
    # 2. 数据准备
    # ----------------------------
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])

    trainset = torchvision.datasets.CIFAR100(
        root='./data', train=True, download=True, transform=transform_train
    )
    testset = torchvision.datasets.CIFAR100(
        root='./data', train=False, download=True, transform=transform_test
    )

    train_loader = DataLoader(trainset, batch_size=64, shuffle=True, num_workers=4)
    test_loader = DataLoader(testset, batch_size=128, shuffle=False, num_workers=4)

    # ----------------------------
    # 3. 模型、优化器、损失函数
    # ----------------------------
    model = torchvision.models.resnet18(pretrained=False, num_classes=100)
    optimizer = AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    # ----------------------------
    # 4. 使用 accelerator.prepare() 包装所有组件
    # ----------------------------
    model, optimizer, train_loader, test_loader = accelerator.prepare(
        model, optimizer, train_loader, test_loader
    )

    # ----------------------------
    # 5. 训练循环
    # ----------------------------
    num_epochs = 10
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        progress_bar = tqdm(train_loader, disable=not accelerator.is_local_main_process)

        for batch in progress_bar:
            images, labels = batch
            with accelerator.accumulate(model):  # 自动处理梯度累积和同步
                outputs = model(images)
                loss = criterion(outputs, labels)
                accelerator.backward(loss)       # 替代 loss.backward()
                optimizer.step()
                optimizer.zero_grad()

            total_loss += loss.item()
            progress_bar.set_description(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}")

        avg_loss = total_loss / len(train_loader)
        accelerator.print(f"Epoch {epoch+1} - Average Train Loss: {avg_loss:.4f}")

        # ----------------------------
        # 6. 验证（仅主进程打印结果）
        # ----------------------------
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in test_loader:
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        # 聚合多卡结果（如果使用分布式）
        correct_tensor = torch.tensor(correct, device=accelerator.device)
        total_tensor = torch.tensor(total, device=accelerator.device)
        correct_tensor = accelerator.gather(correct_tensor).sum()
        total_tensor = accelerator.gather(total_tensor).sum()

        acc = correct_tensor.item() / total_tensor.item()
        accelerator.print(f"Validation Accuracy: {acc:.4f}")

    # ----------------------------
    # 7. 保存模型（仅主进程保存，避免重复）
    # ----------------------------
    accelerator.wait_for_everyone()
    unwrapped_model = accelerator.unwrap_model(model)
    save_path = os.path.join(accelerator.project_dir, "cifar100_resnet18.pth")
    accelerator.save(unwrapped_model.state_dict(), save_path)
    accelerator.print(f"Model saved to {save_path}")

if __name__ == "__main__":
    main()

    """
    单卡训练：
        python train_cifar100_accelerate.py

    多卡训练：
        # 自动配置（交互式）
        accelerate config
        # 或直接运行（自动使用所有可见 GPU）
        accelerate launch train_cifar100_accelerate.py

    """