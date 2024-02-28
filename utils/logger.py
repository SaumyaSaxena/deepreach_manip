from heapq import heappush, heappop

class TopKLogger:
    def __init__(self, k: int):
        self.max_to_keep = k
        self.checkpoint_queue = []
    
    def push(self, ckpt: str, success: float):
        # NOTE: We have a min heap
        if len(self.checkpoint_queue) < self.max_to_keep:
            heappush(self.checkpoint_queue, (success, ckpt))
            return True
        else:
            curr_min_success, _ = self.checkpoint_queue[0]
            if curr_min_success < success:
                heappop(self.checkpoint_queue)
                heappush(self.checkpoint_queue, (success, ckpt))
                return True
            else:
                return False