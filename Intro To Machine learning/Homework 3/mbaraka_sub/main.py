"""Main script for the solution."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse

import npnn


def _get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--lr", help="learning rate", type=float, default=0.1)
    p.add_argument("--opt", help="optimizer", default="SGD")
    p.add_argument(
        "--epochs", help="number of epochs to train", type=int, default=20)
    p.add_argument(
        "--save_stats", help="Save statistics to file", action="store_true")
    p.add_argument(
        "--save_pred", help="Save predictions to file", action="store_true")
    p.add_argument("--dataset", help="Dataset file", default="mnist.npz")
    p.add_argument(
        "--test_dataset", help="Dataset file (test set)",
        default="mnist_test.npz")
    p.set_defaults(save_stats=False, save_pred=False)
    return p.parse_args()


# A function for training 

def train_model(args, lr=None):
    if lr is None:
        lr = args.lr
        
    # Load
    X, y = npnn.load_mnist(args.dataset)

    
    # Flatten images
    if len(X.shape) > 2:
        X = X.reshape(X.shape[0], -1) 
    
    
    p = np.random.permutation(len(y))
    X, y = X[p], y[p]

    train = npnn.Dataset(X[:50000], y[:50000], batch_size=32)
    val = npnn.Dataset(X[50000:], y[50000:], batch_size=32)
    
    # modeling
    in_dim = X.shape[1]  
    model = npnn.Sequential(
        modules=[
            npnn.Dense(dim_in=in_dim, dim_out=256),
            npnn.ELU(alpha=0.9),
            npnn.Dense(256, 64),
            npnn.ELU(alpha=0.9),
            npnn.Dense(64, 10),
        ],
        loss=npnn.SoftmaxCrossEntropy(),
        optimizer=npnn.SGD(learning_rate=lr)
    )

    # Training
    stats_list = []
    train_losses, train_accs = [], []
    val_losses, val_accs = [], []

    for epoch in range(args.epochs):
        # Train one epoch
        train_loss, train_acc = model.train(train)
        # Evaluate on validation set
        val_loss, val_acc = model.test(val)

        print(f"[LR={lr}] Epoch {epoch+1}/{args.epochs} - "
          f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
          f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        
        # Store metrics
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        
        # Store stats in a list to be converted to DataFrame later
        stats_list.append({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc
        })

    # Convert to DataFrame
    stats = pd.DataFrame(stats_list)
    return model, stats, train_losses, train_accs, val_losses, val_accs


if __name__ == '__main__':
    args = _get_args()
    
    # solutin for 6.4.a - Train model with default learning rate
    model, stats, train_losses, train_accs, val_losses, val_accs = train_model(args)
    
    # training curves
    plt.figure(figsize=(12, 5))
    
    # loss curves
    plt.subplot(1, 2, 1)
    plt.plot(range(1, args.epochs + 1), train_losses, label='Train Loss')
    plt.plot(range(1, args.epochs + 1), val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    
    # accuracy curves
    plt.subplot(1, 2, 2)
    plt.plot(range(1, args.epochs + 1), train_accs, label='Train Accuracy')
    plt.plot(range(1, args.epochs + 1), val_accs, label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('training_curves.png')
    plt.show()
    
    # solution for 6.4.b - predictions for test set
    if args.save_pred:
        X_test, _ = npnn.load_mnist(args.test_dataset)
        
        # Flatten test images 
        if len(X_test.shape) > 2:
            X_test = X_test.reshape(X_test.shape[0], -1)
            
        y_pred = np.argmax(model.forward(X_test), axis=1).astype(np.uint8)
        np.save("mnist_test_pred.npy", y_pred)
    
    # solution for 6.4.c - different learning rates
    print("Braaaaaaakaaa")
    if True:  
        print("LETS STATRT HEEEHRJKSEO")
        learning_rates = [0.05, 0.1, 0.2, 0.5, 1.0]
        best_val_losses = []
        
        for lr in learning_rates:
            _, stats, _, _, val_losses, _ = train_model(args, lr=lr)
            best_val_losses.append(min(val_losses))
        print("YOU KNOW QHAAAT")
        # Plot learning rate comparison
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(learning_rates, best_val_losses, 'o-', label='Best Validation Loss')
        ax.set_xscale('log')
        ax.set_xlabel('Learning Rate')
        ax.set_ylabel('Best Validation Loss')
        ax.set_title('Effect of Learning Rate on Validation Loss')
        ax.set_xticks(learning_rates)
        ax.set_xticklabels([str(lr) for lr in learning_rates])
        ax.grid(True)
        ax.legend()

        plt.savefig('lr_comparison.png')
        plt.show()
        print("Experiment completed. Plot saved as 'lr_comparison.png'.")
        # # learning rate comparison
        # plt.figure(figsize=(10, 6))
        # plt.plot(learning_rates, best_val_losses, 'o-')
        # plt.xscale('log')
        # plt.xlabel('Learning Rate')
        # plt.ylabel('Best Validation Loss')
        # plt.title('Learning Rate Effect on Validation Loss')
        # plt.xticks(learning_rates, [str(lr) for lr in learning_rates])
        # plt.grid(True)
        # plt.savefig('lr_comparison.png')
        # plt.show()
    
    
    # Save statistics to file.
    if args.save_stats:
        stats.to_csv(f"stats_{args.opt}_{args.lr}.csv")