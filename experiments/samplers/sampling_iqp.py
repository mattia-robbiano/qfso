from qfso.models.approximate import IQPSampler

dataset = "MNIST"
n_qubits = 28*28
pool_size = 100_000
sigma = 7.2
path = "/Users/mattiaro/repo/qfso/data"
batch_size = 1000
num_batches = 1

iqp_sampler = IQPSampler(
    dataset=dataset,
    n_qubits=n_qubits,
    pool_size=pool_size,
    sigma=sigma,
    path=path,
    num_batches=num_batches,
    batch_size=batch_size,
)
