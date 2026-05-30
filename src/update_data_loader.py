import os
import re

workspace = r"c:\Users\jnagawa\Downloads\hiv-subtype-classifier-20260529T180143Z-3-001\hiv-subtype-classifier"
filepath = os.path.join(workspace, "src", "data_loader.py")

with open(filepath, "r", encoding="utf-8") as f:
    content = f.read()

# Replace block 1 in prepare_data
old_block1 = """    from torch.utils.data import WeightedRandomSampler
    from collections import Counter
    class_counts_dict = Counter(y_train)
    weights_dict = {cls: 1.0 / count for cls, count in class_counts_dict.items()}
    sample_weights = [weights_dict[y] for y in y_train]
    sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(y_train), replacement=True)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler,
                              num_workers=2, pin_memory=True,
                              collate_fn=collate_fn)"""

new_block1 = """    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=2, pin_memory=True,
                              collate_fn=collate_fn)"""

# Replace block 2 in create_dataloaders
old_block2 = """    from torch.utils.data import WeightedRandomSampler
    from collections import Counter
    class_counts_dict = Counter(y_train)
    weights_dict = {cls: 1.0 / count for cls, count in class_counts_dict.items()}
    sample_weights = [weights_dict[y] for y in y_train]
    sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(y_train), replacement=True)

    train_loader = DataLoader(
        HIVSequenceDataset(X_train, y_train, max_length, encoding),
        batch_size=batch_size, sampler=sampler, num_workers=2, pin_memory=True,
        collate_fn=collate_fn
    )"""

new_block2 = """    train_loader = DataLoader(
        HIVSequenceDataset(X_train, y_train, max_length, encoding),
        batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True,
        collate_fn=collate_fn
    )"""

content = content.replace(old_block1, new_block1)
content = content.replace(old_block2, new_block2)

with open(filepath, "w", encoding="utf-8") as f:
    f.write(content)

print("data_loader.py successfully updated!")
