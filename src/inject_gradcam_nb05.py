import json
import os

workspace = r"c:\Users\jnagawa\Downloads\hiv-subtype-classifier-20260529T180143Z-3-001\hiv-subtype-classifier"
nb05_path = os.path.join(workspace, "notebooks", "05_evaluation_and_analysis.ipynb")

with open(nb05_path, "r", encoding="utf-8") as f:
    nb5 = json.load(f)

markdown_cell = {
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "## Model Explainability: High-Confidence Errors (Grad-CAM)\n",
        "\n",
        "If our best model is Convolutional (CNN), we can use **Grad-CAM** to literally \"look inside its brain\" and see exactly which nucleotides caused it to make a high-confidence mistake on the unseen Test Set.\n",
        "\n",
        "This is invaluable for debugging biological edge cases: is the model hyper-fixating on a specific mutation, or is the sequence a recombinant form that genuinely shares features of multiple subtypes?"
    ]
}

code_cell = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "from matplotlib.colors import LinearSegmentedColormap\n",
        "\n",
        "class GradCAM1D:\n",
        "    def __init__(self, model, target_layer):\n",
        "        self.model = model\n",
        "        self.gradients = None\n",
        "        self.activations = None\n",
        "        target_layer.register_forward_hook(self._save_activation)\n",
        "        target_layer.register_full_backward_hook(self._save_gradient)\n",
        "\n",
        "    def _save_activation(self, module, inp, out):\n",
        "        self.activations = out.detach()\n",
        "\n",
        "    def _save_gradient(self, module, grad_in, grad_out):\n",
        "        self.gradients = grad_out[0].detach()\n",
        "\n",
        "    def generate(self, input_tensor, target_class=None):\n",
        "        self.model.eval()\n",
        "        x = input_tensor.to(device).requires_grad_(True)\n",
        "        output = self.model(x)\n",
        "        \n",
        "        probs = F.softmax(output, dim=1)[0].detach().cpu().numpy()\n",
        "        pred_class = output.argmax(dim=1).item()\n",
        "        \n",
        "        if target_class is None:\n",
        "            target_class = pred_class\n",
        "            \n",
        "        self.model.zero_grad()\n",
        "        output[0, target_class].backward()\n",
        "        \n",
        "        weights = self.gradients.mean(dim=2, keepdim=True)\n",
        "        cam = (weights * self.activations).sum(dim=1)[0]\n",
        "        cam = F.relu(cam)\n",
        "        cam = F.interpolate(cam.unsqueeze(0).unsqueeze(0), size=input_tensor.shape[2], mode='linear', align_corners=False)[0, 0].cpu().numpy()\n",
        "        \n",
        "        if cam.max() > 0:\n",
        "            cam = (cam - cam.min()) / (cam.max() - cam.min())\n",
        "        return cam, pred_class, probs\n",
        "\n",
        "def plot_gradcam_1d(cam, sequence, true_class, pred_class, probs, title):\n",
        "    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 4), gridspec_kw={'height_ratios': [1, 3]})\n",
        "    \n",
        "    cmap = LinearSegmentedColormap.from_list('custom', ['#ffffff', '#ff0000'])\n",
        "    im = ax1.imshow(cam.reshape(1, -1), cmap=cmap, aspect='auto')\n",
        "    ax1.set_title(title, fontsize=12, pad=10)\n",
        "    ax1.set_yticks([])\n",
        "    \n",
        "    ax2.plot(cam, color='red', linewidth=1.5)\n",
        "    ax2.fill_between(range(len(cam)), cam, alpha=0.3, color='red')\n",
        "    ax2.set_xlim(0, len(cam))\n",
        "    ax2.set_ylim(0, 1.05)\n",
        "    ax2.set_ylabel('Attention Weight')\n",
        "    ax2.set_xlabel('Nucleotide Position (bp)')\n",
        "    \n",
        "    # Mark top 5% activation regions\n",
        "    threshold = np.percentile(cam, 95)\n",
        "    high_act = np.where(cam >= threshold)[0]\n",
        "    for idx in high_act:\n",
        "        ax2.axvline(x=idx, color='black', alpha=0.1, linewidth=1)\n",
        "        \n",
        "    plt.tight_layout()\n",
        "    return fig\n",
        "\n",
        "# Only run Grad-CAM if the best model is the CNN\n",
        "if 'CNN' in best_name:\n",
        "    print(f\"\\nEvaluating High-Confidence Errors for {best_name} using Grad-CAM...\")\n",
        "    \n",
        "    # Find the single most confident ERROR in the test set\n",
        "    errors_mask = yt != yp\n",
        "    if errors_mask.sum() > 0:\n",
        "        error_indices = np.where(errors_mask)[0]\n",
        "        confidences = yprob[error_indices].max(axis=1)\n",
        "        worst_idx = error_indices[np.argmax(confidences)]\n",
        "        \n",
        "        print(f\"Found worst error at Test Set Index {worst_idx}\")\n",
        "        true_lbl = class_names[yt[worst_idx]]\n",
        "        pred_lbl = class_names[yp[worst_idx]]\n",
        "        conf = yprob[worst_idx].max() * 100\n",
        "        print(f\"True Subtype: {true_lbl} | Predicted: {pred_lbl} (Confidence: {conf:.1f}%)\")\n",
        "        \n",
        "        # Fetch sequence and tensor\n",
        "        seq = X_test[worst_idx]\n",
        "        input_tensor = next(iter(test_dl_oh))[0][worst_idx:worst_idx+1].to(device) # Get exactly this sequence encoded\n",
        "        # Note: Dataloader shuffle=False for test, so indices match perfectly.\n",
        "        \n",
        "        # Hook into the final Convolutional layer of the CNN\n",
        "        # The conv block is a Sequential. We grab the last Conv1d layer.\n",
        "        target_layer = cnn.conv[17] # Conv1d(256,256)\n",
        "        \n",
        "        gradcam = GradCAM1D(cnn, target_layer)\n",
        "        cam, _, _ = gradcam.generate(input_tensor, target_class=yp[worst_idx])\n",
        "        \n",
        "        title = f\"Grad-CAM: Why did {best_name} predict {pred_lbl} (instead of {true_lbl}) with {conf:.1f}% confidence?\"\n",
        "        fig = plot_gradcam_1d(cam, seq, true_lbl, pred_lbl, probs=None, title=title)\n",
        "        fig.savefig(f'{reports}/gradcam_worst_error.png', dpi=150, bbox_inches='tight')\n",
        "        plt.show()\n",
        "    else:\n",
        "        print(\"The model made absolutely no errors on the test set! (Incredible!)\")\n",
        "else:\n",
        "    print(f\"\\nThe best model is {best_name}, which is not a CNN. Grad-CAM is optimized for Convolutional layers, skipping explainability.\")\n"
    ]
}

nb5['cells'].append(markdown_cell)
nb5['cells'].append(code_cell)

with open(nb05_path, "w", encoding="utf-8") as f:
    json.dump(nb5, f, indent=1)

print("Grad-CAM successfully added to Notebook 05!")
