import itertools as it
import numpy as np

C, T = 8, 4
a = np.fabs(np.random.randn(T, C))

# brute
mn = np.inf
mn_n = None
for n in it.product(range(C), repeat=T):
    if (np.array(n)+1).sum() != C: continue
    if mn > a[np.arange(T), n].max():
        mn = a[np.arange(T), n].max()
        mn_n = n

print('brute:')
print(mn_n, mn)

# dynprog

# dp = np.empty((T, C))
# dp[0] = a[0]
dp = {}
for c in range(1,C+1): dp[1,c] = a[0,c-1]

for t in range(1,T):
    for c in range(t+1,C+1):
        subprobs = []
        for nt in range(c-t):
            subprobs.append( max(a[t, nt], dp[t, c-nt-1]) )
        dp[t+1, c] = min(subprobs)

print('dynprog:')
print(dp[T, C])

assert dp[T,C] == mn
