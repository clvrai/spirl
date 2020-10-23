import torch
import numpy as np
import os


def save_checkpoint(state, filename='checkpoint.pth'):
    torch.save(state, filename)


class BaseTrainer():
    def override_defaults(self, policyparams):
        for name, value in policyparams.items():
            print('overriding param {} to value {}'.format(name, value))
            if value == getattr(self._hp, name):
                raise ValueError("attribute is {} is identical to default value!!".format(name))
            self._hp.set_hparam(name, value)

    def call_hooks(self, inputs, output, losses, epoch):
        for hook in self.hooks:
            hook.run(inputs, output, losses, epoch)

    def check_nan_grads(self, grads):
        grads = grads #or self.model.named_parameters()
        return torch.stack([torch.isnan(p.grad).any() for n, p in grads]).any()

    def nan_grads_hook(self, inputs, output, losses, epoch):
        non_none = list(filter(lambda x: x[1].grad is not None, self.model.named_parameters()))
        if self.check_nan_grads(non_none):
            self.dump_debug_data(inputs, output, losses, epoch)
            nan_parameters = [n for n,v in filter(lambda x: torch.isnan(x[-1].grad).any(), non_none)]
            non_nan_parameters = [n for n, v in filter(lambda x: not torch.isnan(x[-1].grad).any(), non_none)]
            # grad_dict = self.print_nan_grads()
            print('nan gradients here: ', nan_parameters)
            print('there parameters are OK: ', non_nan_parameters)

            def get_nan_module(layer):
                return np.unique(map(lambda n: '.'.join(n.split('.')[:layer]), nan_parameters))

            import pdb; pdb.set_trace()
            raise ValueError('there are NaN Gradients')
        
    def dump_debug_data(self, inputs, outputs, losses, epoch):
        print("Detected NaN gradients, dumping data for diagnosis, ...")
        debug_dict = dict()

        def clean_dict(d):
            clean_d = dict()
            for key, val in d.items():
                if isinstance(val, torch.Tensor):
                    clean_d[key] = val.data.cpu().numpy()
                elif isinstance(val, dict):
                    clean_d[key] = clean_dict(val)
            return clean_d

        debug_dict['inputs'] = clean_dict(inputs)
        debug_dict['outputs'] = clean_dict(outputs)
        debug_dict['losses'] = clean_dict(losses)

        import pickle as pkl
        f = open(os.path.join(self._hp.exp_path, "nan_debug_info.pkl"), "wb")
        pkl.dump(debug_dict, f)
        f.close()

        save_checkpoint({
            'epoch': epoch,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
        }, os.path.join(self._hp.exp_path, "nan_debug_ckpt.pth"))

    def print_nan_grads(self):
        grad_dict = {}
        for name, param in self.model.named_parameters(recurse=True):
            print("{}:\t\t{}".format(name, bool(torch.isnan(param.grad).any().data.cpu().numpy())))
            grad_dict[name] = param.grad
        return grad_dict
