import torch

torch.set_printoptions(precision=10, threshold=float('inf'))
data = torch.load("./data/card_set/train.pt")

print(data[0:2])

count = 0
for item in data:
    # Check if the item has indexable structure
    if hasattr(item, "__getitem__"):
        try:
            if item[-2] == 4:
                count += 1
        except (IndexError, TypeError):
            print("Item doesn't have a penultimate element or isn't indexable as expected")
            break

print(f"Count of items with value 4 at penultimate position: {count}")
print(f"Percentage: {(count / len(data)) * 100:.2f}%")