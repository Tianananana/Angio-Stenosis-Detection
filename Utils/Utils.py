import torch
from torchvision.ops import box_area


def collate(batch):
    """
    Collate function for dataloader
    """
    return list(zip(*batch))


class PostProcess:
    """
    Post processing utils
    """
    @staticmethod
    def intersect(box1, box2):
        """
        returns bbox of intersection between two boxes of shape (1, 4)
        """
        x = [box1[0], box1[2], box2[0], box2[2]]
        y = [box1[1], box1[3], box2[1], box2[3]]
        x.sort()
        y.sort()
        xmin, ymin, xmax, ymax = x[1], y[1], x[2], y[2]
        box = torch.Tensor([xmin, ymin, xmax, ymax])
        box = box[None, :]
        # box = torch.unsqueeze(box, dim=0)
        return box

    @staticmethod
    def IoA(boxes):
        """
        param boxes: list of predicted boxes
        returns pairwise Intersection-over-area of shape (N, N)
        """
        n = boxes.size()[0]
        idx = torch.arange(n)
        i, j = torch.meshgrid(idx, idx)
        idx_i, idx_j = boxes[i, :], boxes[j, :]

        IoA = torch.empty((n, n))
        for x in range(n):
            for y in range(n):
                b1, b2 = idx_i[x, y, :], idx_j[x, y, :]
                IoA[x, y] = box_area(PostProcess.intersect(b1, b2)) / box_area(torch.unsqueeze(b1, dim=0))
        return IoA

    @staticmethod
    def IoA2(boxes1, boxes2):
        """
        param boxes1, boxes2: list of two boxes of length M, N for comparison
        returns pairwise Intersection-over area of shape (M, N)
        """
        m, n = boxes1.shape[0], boxes2.shape[0]
        intersection = torch.empty((m, n))
        for i in range(m):
            for j in range(n):
                b1, b2 = boxes1[i], boxes2[j]
                intersection[i, j] = box_area(PostProcess.intersect(b1, b2)) / box_area(torch.unsqueeze(b1, dim=0))

        return intersection

def eager_outputs(self, losses, detections):
    # type: (Dict[str, Tensor], List[Dict[str, Tensor]]) -> Union[Dict[str, Tensor], List[Dict[str, Tensor]]]
    if self.training:
        return losses
    return losses, detections