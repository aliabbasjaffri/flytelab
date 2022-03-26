import torch
from torch import nn
from typing import List, Dict
from torch.utils.data import DataLoader
from model import MultiTemporalFeatureMap
from data_split import generate_data_split
from utils import get_default_device, to_device
from torchaudio.datasets.gtzan import gtzan_genres


@torch.no_grad()
def evaluate(model: MultiTemporalFeatureMap, test_dataloader: DataLoader):
    model.eval()
    outputs = [model.validation_step(batch) for batch in test_dataloader]
    return model.validation_epoch_end(outputs)


def get_learning_rate(optimizer):
    for param_group in optimizer.param_groups:
        return param_group["lr"]


def train(
    epochs: int,
    max_learning_rate: float,
    training_dataloader: DataLoader,
    validation_dataloader: DataLoader,
    optimization_function=torch.optim.Adam,
    weight_decay: float = 0.0,
    gradient_clip: float = None,
) -> (List[any], MultiTemporalFeatureMap):
    torch.cuda.empty_cache()
    history = []

    model = to_device(
        data=MultiTemporalFeatureMap(in_channels=1, num_classes=len(gtzan_genres)),
        device=get_default_device(),
    )
    print(model)

    # Set up custom optimizer with weight decay
    optimizer = optimization_function(
        model.parameters(), max_learning_rate, weight_decay=weight_decay
    )
    # Set up one-cycle learning rate scheduler
    schedule = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_learning_rate,
        epochs=epochs,
        steps_per_epoch=len(training_dataloader),
    )

    for epoch in range(epochs):
        # Training Phase
        model.train()
        train_losses = []
        lrs = []
        for batch in training_dataloader:
            loss = model.training_step(batch)
            train_losses.append(loss)
            loss.backward()

            # Gradient clipping
            if gradient_clip:
                nn.utils.clip_grad_value_(model.parameters(), gradient_clip)

            optimizer.step()
            optimizer.zero_grad()

            # Record & update learning rate
            lrs.append(get_learning_rate(optimizer))
            schedule.step()

        # Validation phase
        result = evaluate(model, validation_dataloader)
        result["train_loss"] = torch.stack(train_losses).mean().item()
        result["lrs"] = lrs
        model.epoch_end(epoch, result)
        print(result)
        history.append(result)
    # add history to mlflow
    return history, model


if __name__ == "__main__":
    history = []

    train_dataloader, validation_dataloader, test_dataloader = generate_data_split(
        train_split_pc=0.01, validation_split_pc=0.01, test_split_pc=0.98
    )

    print("data splits generated")

    epochs = 1  # 20
    max_lr = 0.01
    grad_clip = 0.1
    weight_decay = 1e-4
    opt_func = torch.optim.Adam

    _history, model = train(
        epochs=epochs,
        max_learning_rate=max_lr,
        training_dataloader=train_dataloader,
        validation_dataloader=validation_dataloader,
        gradient_clip=grad_clip,
        weight_decay=weight_decay,
        optimization_function=opt_func,
    )

    _eval_result = evaluate(model, validation_dataloader)
    history.append(_eval_result)

    history.append(_history)
    print(history)
