# Q51.) N-Queens

# The n-queens puzzle is the problem of placing n queens on an n x n chessboard such that no two queens attack each other.

# Given an integer n, return all distinct solutions to the n-queens puzzle. You may return the answer in any order.

# Each solution contains a distinct board configuration of the n-queens' placement, where 'Q' and '.' both indicate a queen and an empty space, respectively.

# Example 1:


# Input: n = 4
# Output: [[".Q..","...Q","Q...","..Q."],["..Q.","Q...","...Q",".Q.."]]
# Explanation: There exist two distinct solutions to the 4-queens puzzle as shown above
# Example 2:

# Input: n = 1
# Output: [["Q"]]
 
# Constraints:

# 1 <= n <= 9

# Sol_51}

class Solution:
    def solve(self, col, board, ans, leftrow, upperDiagonal, lowerDiagonal, n):
        if col == n:
            ans.append(board[:])
            return

        for row in range(n):
            if (
                leftrow[row] == 0
                and lowerDiagonal[row + col] == 0
                and upperDiagonal[n - 1 + col - row] == 0
            ):
                board[row] = board[row][:col] + "Q" + board[row][col + 1 :]
                leftrow[row] = 1
                lowerDiagonal[row + col] = 1
                upperDiagonal[n - 1 + col - row] = 1
                self.solve(col + 1, board, ans, leftrow, upperDiagonal, lowerDiagonal, n)
                board[row] = board[row][:col] + "." + board[row][col + 1 :]
                leftrow[row] = 0
                lowerDiagonal[row + col] = 0
                upperDiagonal[n - 1 + col - row] = 0

    def solveNQueens(self, n: int) -> List[List[str]]:
        ans = []
        board = ["." * n for _ in range(n)]
        leftrow = [0] * n
        upperDiagonal = [0] * (2 * n - 1)
        lowerDiagonal = [0] * (2 * n - 1)
        self.solve(0, board, ans, leftrow, upperDiagonal, lowerDiagonal, n)
        return ans
    
# Q52.) N-Queens II

# The n-queens puzzle is the problem of placing n queens on an n x n chessboard such that no two queens attack each other.

# Given an integer n, return the number of distinct solutions to the n-queens puzzle.

# Example 1:

# Input: n = 4
# Output: 2
# Explanation: There are two distinct solutions to the 4-queens puzzle as shown.
# Example 2:

# Input: n = 1
# Output: 1
 
# Constraints:
# 1 <= n <= 9

# Sol_52}

class Solution:
    def totalNQueens(self, n: int) -> int:
        if n == 1:
            return 1
        
        def get_atack_map(i, j):
            atack_map = {k: set() for k in range(i + 1, n)}
            for ii in range(i + 1, n):
                atack_map[ii].add(j)
                
            for shift in range(1, min(n - i, n - j)):
                atack_map[i+shift].add(j + shift)
            
            for shift in range(1, min(j + 1, n - i)):
                atack_map[i + shift].add(j - shift)
            
            return atack_map
        
        initial_set = set(range(n))
        stack = [(0, {k: initial_set for k in range(n)})]
        count = 0

        while stack:
            line, valid_map = stack.pop()
            if line == n - 1: 
                if valid_map[n - 1]:
                    count += 1
                else:
                    continue
            
            else:
                for pos in valid_map[line]:
                    atack_map = get_atack_map(line, pos)
                    valid_map_next = {k: valid_map[k] - atack_map[k]  for k in range(line + 1,n)}
                    stack.append((line + 1, valid_map_next))
        
        return count 

# Q53.) Maximum Subarray
# Given an integer array nums, find the subarray with the largest sum, and return its sum.

# Example 1:

# Input: nums = [-2,1,-3,4,-1,2,1,-5,4]
# Output: 6
# Explanation: The subarray [4,-1,2,1] has the largest sum 6.
# Example 2:

# Input: nums = [1]
# Output: 1
# Explanation: The subarray [1] has the largest sum 1.
# Example 3:

# Input: nums = [5,4,-1,7,8]
# Output: 23
# Explanation: The subarray [5,4,-1,7,8] has the largest sum 23.
 
# Constraints:

# 1 <= nums.length <= 105
# -104 <= nums[i] <= 104
 
# Follow up: If you have figured out the O(n) solution, try coding another solution using the divide and conquer approach, which is more subtle.

# Sol_53}

class Solution:
    def maxSubArray(self, nums: List[int]) -> int:
        res = nums[0]
        total = 0

        for n in nums:
            if total < 0:
                total = 0

            total += n
            res = max(res, total)
        
        return res       