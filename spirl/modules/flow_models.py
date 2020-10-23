import torch
import torch.nn as nn
import numpy as np

from spirl.modules.subnetworks import Predictor
from spirl.modules.variational_inference import MultivariateGaussian


class NormalizingFlowModel(nn.Module):
    """
    Joins multiple flow models into composite flow.
    Implementation extended from: https://github.com/tonyduan/normalizing-flows/blob/master/nf/models.py
    """

    def __init__(self, flow_dim, flows):
        super().__init__()
        self._flow_dim = flow_dim
        self.flows = nn.ModuleList(flows)

    def forward(self, x, cond_inputs=None):
        m, _ = x.shape
        log_det = torch.zeros(m, device=x.device)
        for flow in self.flows:
            x, ld = flow.forward(x, cond_inputs)
            log_det += ld
        z, prior_logprob = x, self._get_prior(m, x.device).log_prob(x)
        return z, prior_logprob, log_det

    def inverse(self, z, cond_inputs=None):
        m, _ = z.shape
        log_det = torch.zeros(m, device=z.device)
        for flow in self.flows[::-1]:
            z, ld = flow.inverse(z, cond_inputs)
            log_det += ld
        x = z
        return x, log_det

    def sample(self, num_samples=None, device=None, cond_inputs=None):
        if num_samples is None:
            num_samples = cond_inputs[0].shape[0]
        if device is None:
            device = cond_inputs[0].device
        z = self._get_prior(batch_size=num_samples, device=device).sample()
        x, _ = self.inverse(z, cond_inputs)
        return x

    def _get_prior(self, batch_size, device):
        return MultivariateGaussian(torch.zeros((batch_size, self._flow_dim), requires_grad=False, device=device),
                                    torch.zeros((batch_size, self._flow_dim), requires_grad=False, device=device))


class RealNVP(nn.Module):
    """
    Non-volume preserving flow.
    [Dinh et. al. 2017]
    Implementation extended from: https://github.com/tonyduan/normalizing-flows/blob/master/nf/flows.py
    """
    def __init__(self, dim, cond_dim=None, hidden_dim=32):
        """Constructs RealNVP flow. Note that input_dim == output_dim == dim.
           cond_dim allows to add conditioning to the flow model.
        """
        super().__init__()
        assert dim % 2 == 0     # need even input/output dim to use split-in-half scheme
        self.dim = dim
        self.cond_dim = cond_dim
        input_dim = self.dim // 2 if cond_dim is None else self.dim // 2 + cond_dim
        self.t1 = FCNN(in_dim=input_dim, out_dim=dim // 2, hidden_dim=hidden_dim)
        self.s1 = FCNN(in_dim=input_dim, out_dim=dim // 2, hidden_dim=hidden_dim)
        self.t2 = FCNN(in_dim=input_dim, out_dim=dim // 2, hidden_dim=hidden_dim)
        self.s2 = FCNN(in_dim=input_dim, out_dim=dim // 2, hidden_dim=hidden_dim)

    def forward(self, x, cond_inputs=None):
        """Forward pass of the RealNVP module. Cond_inputs is a list of conditioning tensors."""
        assert len(x.shape) == 2 and x.shape[-1] == self.dim
        if cond_inputs is not None:
            assert np.prod([ci.shape[-1] for ci in cond_inputs]) == self.cond_dim
        lower, upper = x[:, :self.dim // 2], x[:, self.dim // 2:]
        t1_transformed = self.t1(lower, cond_inputs)
        s1_transformed = self.s1(lower, cond_inputs)
        upper = t1_transformed + upper * torch.exp(s1_transformed)
        t2_transformed = self.t2(upper, cond_inputs)
        s2_transformed = self.s2(upper, cond_inputs)
        lower = t2_transformed + lower * torch.exp(s2_transformed)
        z = torch.cat([lower, upper], dim=1)
        log_det = torch.sum(s1_transformed, dim=1) + \
                  torch.sum(s2_transformed, dim=1)
        return z, log_det

    def inverse(self, z, cond_inputs=None):
        assert len(z.shape) == 2 and z.shape[-1] == self.dim
        if cond_inputs is not None:
            assert np.prod([ci.shape[-1] for ci in cond_inputs]) == self.cond_dim
        lower, upper = z[:, :self.dim // 2], z[:, self.dim // 2:]
        t2_transformed = self.t2(upper, cond_inputs)
        s2_transformed = self.s2(upper, cond_inputs)
        lower = (lower - t2_transformed) * torch.exp(-s2_transformed)
        t1_transformed = self.t1(lower, cond_inputs)
        s1_transformed = self.s1(lower, cond_inputs)
        upper = (upper - t1_transformed) * torch.exp(-s1_transformed)
        x = torch.cat([lower, upper], dim=1)
        log_det = torch.sum(-s1_transformed, dim=1) + \
                  torch.sum(-s2_transformed, dim=1)
        return x, log_det


class FCNN(nn.Module):
    """
    Simple fully connected neural network.
    """
    def __init__(self, in_dim, out_dim, hidden_dim):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x, additional_inputs):
        input = torch.cat([x] + additional_inputs, dim=-1) if additional_inputs is not None else x
        return self.network(input)


class FlowDistributionWrapper:
    """Lightweight wrapper around flow model that makes it behave like distribution."""
    def __init__(self, flow, cond_inputs=None):
        self._flow = flow
        self._cond_inputs = cond_inputs
        self._detached = False      # indicates whether flow output should be detached

    def log_prob(self, x):
        _, prior_logprob, log_det = self._flow(x, self._cond_inputs)
        if self._detached:
            prior_logprob, log_det = prior_logprob.detach(), log_det.detach()
        return prior_logprob + log_det

    def nll(self, x):
        return -1 * self.log_prob(x)

    def sample(self):
        return self._flow.sample(cond_inputs=self._cond_inputs)

    def rsample(self):
        return self.sample()

    @staticmethod
    def cat(*argv, dim):
        # TODO: implement concatentation for flow distribution
        return argv[0]

    def detach(self):
        self._detached = True
        return self

    def to_numpy(self):
        return np.zeros((1,))   # there is no numpy representation for an implicit function

    def entropy(self):
        return np.array(0.)  # dummy value - entropy of flow not defined


class ConditionedFlowModel(nn.Module):
    """Wraps flow model and conditioning network."""
    def __init__(self, hp, input_dim, output_dim, n_flow_layers):
        super().__init__()
        self._hp = hp
        self._cond_net = Predictor(hp, input_size=input_dim, output_size=self._hp.nz_mid_prior,
                                   num_layers=self._hp.num_prior_net_layers, mid_size=self._hp.nz_mid_prior)
        self._flows = [RealNVP(output_dim, cond_dim=self._hp.nz_mid_prior) for _ in range(n_flow_layers)]
        self._flow_mdl = NormalizingFlowModel(output_dim, self._flows)

    def forward(self, obs):
        cond = self._cond_net(obs)
        return FlowDistributionWrapper(self._flow_mdl, cond_inputs=[cond])


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # generate data
    data = np.concatenate((np.random.normal(loc=(1.0, 0.0), scale=(0.1, 0.1), size=(1000, 2)),
                           np.random.normal(loc=(-1.0, 0.0), scale=(0.1, 0.1), size=(1000, 2)),
                           np.random.normal(loc=(0.0, 1.0), scale=(0.1, 0.1), size=(1000, 2)),
                           np.random.normal(loc=(0.0, -1.0), scale=(0.1, 0.1), size=(1000, 2))))
    np.random.shuffle(data)

    # set up flow model
    flows = [RealNVP(2) for _ in range(3)]
    model = NormalizingFlowModel(2, flows)

    pydata = torch.tensor(data, dtype=torch.float32)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)

    # train flow model
    for i in range(600):
        optimizer.zero_grad()
        flow_dist = FlowDistributionWrapper(model)
        loss = flow_dist.nll(pydata).mean()
        loss.backward()
        optimizer.step()
        if i % 100 == 0:
            print(f"Iter: {i}\t" +
                  f"NLL: {loss.mean().data:.2f}\t")

    # visualize samples
    samples = flow_dist._flow.sample(num_samples=data.shape[0], device="cpu").data.numpy()
    fig = plt.figure()
    plt.scatter(data[:, 0], data[:, 1], c='black', alpha=0.5)
    plt.scatter(samples[:, 0], samples[:, 1], c='green', alpha=0.5)
    plt.axis("equal")
    plt.savefig("flow_data_fit.png")
    plt.close(fig)


    ### Train second model to fit first model by minimizing empirical KL
    flows2 = [RealNVP(2) for _ in range(3)]
    sample_train_model = NormalizingFlowModel(2, flows2)
    optimizer = torch.optim.Adam(sample_train_model.parameters(), lr=0.005)

    for i in range(10000):
        optimizer.zero_grad()
        # flow_dist = FlowDistributionWrapper(model)
        flow_dist_sample_train = FlowDistributionWrapper(sample_train_model)
        loss_samples = []
        for _ in range(1):
            # data_sample = flow_dist._flow.sample(num_samples=data.shape[0], device="cpu").detach()
            flow_sample = flow_dist_sample_train._flow.sample(num_samples=data.shape[0], device="cpu")
            # loss = flow_dist_sample_train.nll(data_sample).mean()
            loss = (flow_dist_sample_train.log_prob(flow_sample) - flow_dist.log_prob(flow_sample))
            # loss = (flow_dist.log_prob(data_sample) - flow_dist_sample_train.log_prob(data_sample))
            loss_samples.append(loss)
        loss = torch.cat(loss_samples).mean()
        loss.backward()
        optimizer.step()
        if i % 100 == 0:
            print(f"Iter: {i}\t" +
                  f"NLL: {loss.mean().data:.2f}\t")

    # visualize samples
    samples = flow_dist._flow.sample(num_samples=data.shape[0], device="cpu").data.numpy()
    samples_sample_train = flow_dist_sample_train._flow.sample(num_samples=data.shape[0], device="cpu").data.numpy()
    fig = plt.figure()
    plt.scatter(samples[:, 0], samples[:, 1], c='black', alpha=0.5)
    plt.scatter(samples_sample_train[:, 0], samples_sample_train[:, 1], c='green', alpha=0.5)
    plt.axis("equal")
    plt.savefig("flow_sample_fit.png")
    plt.close(fig)
