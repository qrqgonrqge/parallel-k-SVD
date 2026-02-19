import numpy, torch
import juliacall; jl = juliacall.Main
jl.seval("""
    using KSVD
""")

Y = torch.rand(128, 5_000, dtype=torch.float32)
# numpy arrays are understood by Julia
res = jl.ksvd(Y.numpy(), 256, 3)  # m=256, k=3
print(res.D, res.X)

# Convert back to PyTorch tensors
Dtorch = torch.from_numpy(numpy.array(res.D))
Xtorch = torch.sparse_csc_tensor(res.X.colptr, res.X.rowval, res.X.nzval, 
                                 size=(res.X.m, res.X.n), dtype=torch.float32)
