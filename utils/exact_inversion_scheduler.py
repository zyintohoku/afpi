from torch.optim.lr_scheduler import ReduceLROnPlateau
class StepScheduler(ReduceLROnPlateau):
    def __init__(self, mode='min', current_lr=0, factor=0.1, patience=10,
                 threshold=1e-4, threshold_mode='rel', cooldown=0,
                 min_lr=0, eps=1e-8, verbose=False):
        if factor >= 1.0:
            raise ValueError('Factor should be < 1.0.')
        self.factor = factor
        if current_lr == 0:
            raise ValueError('Step size cannot be 0')

        self.min_lr = min_lr
        self.current_lr = current_lr
        self.patience = patience
        self.verbose = verbose
        self.cooldown = cooldown
        self.cooldown_counter = 0
        self.mode = mode
        self.threshold = threshold
        self.threshold_mode = threshold_mode
        self.best = None
        self.num_bad_epochs = None
        self.mode_worse = None  # the worse value for the chosen mode
        self.eps = eps
        self.last_epoch = 0
        self._init_is_better(mode=mode, threshold=threshold,
                             threshold_mode=threshold_mode)
        self._reset()

    def step(self, metrics, epoch=None):
        # convert `metrics` to float, in case it's a zero-dim Tensor
        current = float(metrics)
        if epoch is None:
            epoch = self.last_epoch + 1
        else:
            import warnings
            warnings.warn("EPOCH_DEPRECATION_WARNING", UserWarning)
        self.last_epoch = epoch

        if self.is_better(current, self.best):
            self.best = current
            self.num_bad_epochs = 0
        else:
            self.num_bad_epochs += 1

        if self.in_cooldown:
            self.cooldown_counter -= 1
            self.num_bad_epochs = 0  # ignore any bad epochs in cooldown

        if self.num_bad_epochs > self.patience:
            self._reduce_lr(epoch)
            self.cooldown_counter = self.cooldown
            self.num_bad_epochs = 0

        return self.current_lr

    def _reduce_lr(self, epoch):
        old_lr = self.current_lr
        new_lr = max(self.current_lr * self.factor, self.min_lr)
        if old_lr - new_lr > self.eps:
            self.current_lr = new_lr
            if self.verbose:
                epoch_str = ("%.2f" if isinstance(epoch, float) else
                            "%.5d") % epoch
                print('Epoch {}: reducing learning rate'
                        ' to {:.4e}.'.format(epoch_str,new_lr))
