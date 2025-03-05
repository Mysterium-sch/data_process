import numpy as np
import concurrent.futures

def dense_map(Pts, n, m, grid):
    ng = 2 * grid + 1
    
    # Initialize arrays with inf values
    mX = np.full((m, n), np.inf)
    mY = np.full((m, n), np.inf)
    mD = np.zeros((m, n))
    
    # Fill the initial positions
    mX[np.int32(Pts[1]), np.int32(Pts[0])] = Pts[0] - np.round(Pts[0])
    mY[np.int32(Pts[1]), np.int32(Pts[0])] = Pts[1] - np.round(Pts[1])
    mD[np.int32(Pts[1]), np.int32(Pts[0])] = Pts[2]
    
    # Compute the slices only once
    KmX = np.empty((ng, ng, m - ng, n - ng))
    KmY = np.empty_like(KmX)
    KmD = np.empty_like(KmX)

    # Vectorize S and Y computation
    S = np.zeros((m - ng, n - ng))
    Y = np.zeros((m - ng, n - ng))



    def process_chunk(i, j):
        # Process the chunk in parallel
        s = 1 / np.sqrt((mX[i:(m - ng + i), j:(n - ng + j)] - grid - 1 + i)**2 + (mY[i:(m - ng + i), j:(n - ng + j)] - grid - 1 + j)**2)
        Y_chunk = s * mD[i:(m - ng + i), j:(n - ng + j)]
        return Y_chunk, s

    # Using ThreadPoolExecutor or ProcessPoolExecutor for parallelism
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [executor.submit(process_chunk, i, j) for i in range(ng) for j in range(ng)]
        for future in concurrent.futures.as_completed(futures):
            Y_chunk, S_chunk = future.result()
            Y += Y_chunk
            S += S_chunk

    
    # Avoid division by zero by replacing zeros in S with ones
    S[S == 0] = 1
    
    # Return the output
    out = np.zeros((m, n))
    out[grid + 1:-grid, grid + 1:-grid] = Y / S
    return out
