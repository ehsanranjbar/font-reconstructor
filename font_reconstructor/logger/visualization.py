import importlib
from datetime import datetime


class TensorboardWriter():
    def __init__(self, log_dir, logger, enabled):
        self.writer = None
        self.selected_module = ""

        if enabled:
            log_dir = str(log_dir)

            # Retrieve vizualization writer.
            for module in ["torch.utils.tensorboard", "tensorboardX"]:
                try:
                    self.writer = importlib.import_module(module).SummaryWriter(log_dir)
                    self.selected_module = module
                    break
                except ImportError:
                    continue

            if self.writer is None:
                message = "Warning: visualization (Tensorboard) is configured to use, but currently not installed on " \
                    "this machine. Please install TensorboardX with 'pip install tensorboardx', upgrade PyTorch to " \
                    "version >= 1.1 to use 'torch.utils.tensorboard' or turn off the option in the 'config.json' file."
                logger.warning(message)

        self.step = 0

        # writer functions that receive the current step as their third positional argument
        self.tb_writer_ftns = {
            'add_scalar', 'add_scalars', 'add_image', 'add_images', 'add_audio',
            'add_text', 'add_histogram', 'add_pr_curve', 'add_figure',
        }
        self.timer = datetime.now()

    @property
    def enabled(self):
        return self.writer is not None

    def set_step(self, step):
        self.step = step
        if step == 1:
            self.timer = datetime.now()
        else:
            duration = datetime.now() - self.timer
            self.add_scalar('epoch_time', duration.total_seconds())
            self.timer = datetime.now()

    def __getattr__(self, name):
        """
        If visualization is configured to use:
            return add_data() methods of tensorboard with additional information (step, tag) added.
        Otherwise:
            return a blank function handle that does nothing
        """
        if name.startswith('__'):
            raise AttributeError(name)

        if self.writer is None:
            def noop(*args, **kwargs):
                return None
            return noop

        if name in self.tb_writer_ftns:
            add_data = getattr(self.writer, name)

            def wrapper(tag, data, *args, **kwargs):
                add_data(tag, data, self.step, *args, **kwargs)
            return wrapper

        try:
            return getattr(self.writer, name)
        except AttributeError:
            raise AttributeError("type object '{}' has no attribute '{}'".format(self.selected_module, name))
