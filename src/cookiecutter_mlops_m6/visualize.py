import matplotlib.pyplot as plt
import torch
import typer
from cookiecutter_mlops_m6.model import Model
from cookiecutter_mlops_m6.data import MyDataset
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


def visualize(
    model_checkpoint: str = "./models/model.pth",
    figure_name: str = "embeddings.png",
) -> None:
    """Visualize model predictions."""
    model: torch.nn.Module = Model()
    model.load_state_dict(torch.load(model_checkpoint))
    model.eval()

    _, test_set = MyDataset()
    test_dataloader = torch.utils.data.DataLoader(test_set, batch_size=32)

    embeddings, targets = [], []
    with torch.inference_mode():
        for batch in test_dataloader:
            img, target = batch
            predictions = model(img)
            embeddings.append(predictions)
            targets.append(target)
        embeddings = torch.cat(embeddings).numpy()
        targets = torch.cat(targets).numpy()

    if embeddings.shape[1] > 500:  # Reduce dimensionality for large embeddings
        pca = PCA(n_components=100)
        embeddings = pca.fit_transform(embeddings)
    tsne = TSNE(n_components=2)
    embeddings = tsne.fit_transform(embeddings)

    plt.figure(figsize=(10, 10))
    for i in range(10):
        mask = targets == i
        plt.scatter(
            embeddings[mask, 0],
            embeddings[mask, 1],
            label=str(i),
        )
    plt.legend()
    plt.savefig(f"reports/figures/{figure_name}")


if __name__ == "__main__":
    typer.run(visualize)
