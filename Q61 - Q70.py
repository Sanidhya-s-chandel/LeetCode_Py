# Q61.) Rotate List

# Given the head of a linked list, rotate the list to the right by k places.

# Example 1:


# Input: head = [1,2,3,4,5], k = 2
# Output: [4,5,1,2,3]
# Example 2:

# Input: head = [0,1,2], k = 4
# Output: [2,0,1]
 
# Constraints:

# The number of nodes in the list is in the range [0, 500].
# -100 <= Node.val <= 100
# 0 <= k <= 2 * 109

# Sol_61}

# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution:
    def rotateRight(self, head: ListNode, k: int) -> ListNode:
        
        if not head or not head.next or k == 0:
            return head
        
        # Find the length of the list and the last node
        lastElement = head
        length = 1
        while lastElement.next:
            lastElement = lastElement.next
            length += 1
        
        # Adjust k to be within the range of the list length
        k = k % length
        if k == 0:
            return head
        
        # Connect the last node to the head, forming a circular list
        lastElement.next = head
        
        # Find the new tail (length - k - 1) and new head (length - k)
        new_tail = head
        for _ in range(length - k - 1):
            new_tail = new_tail.next
        
        new_head = new_tail.next
        new_tail.next = None  # Break the circle
        
        return new_head

# Q62.) Unique Paths

# There is a robot on an m x n grid. The robot is initially located at the top-left corner (i.e., grid[0][0]). The robot tries to move to the bottom-right corner (i.e., grid[m - 1][n - 1]). The robot can only move either down or right at any point in time.

# Given the two integers m and n, return the number of possible unique paths that the robot can take to reach the bottom-right corner.

# The test cases are generated so that the answer will be less than or equal to 2 * 109.

# Example 1:

# Input: m = 3, n = 7
# Output: 28
# Example 2:

# Input: m = 3, n = 2
# Output: 3
# Explanation: From the top-left corner, there are a total of 3 ways to reach the bottom-right corner:
# 1. Right -> Down -> Down
# 2. Down -> Down -> Right
# 3. Down -> Right -> Down
 
# Constraints:

# 1 <= m, n <= 100

# Sol_62}

class Solution:
    def uniquePaths(self, m: int, n: int) -> int:
        def dfs(i, j):
            if i > m - 1 or j > n - 1:
                return 0
            
            if memo[i][j] != -1:
                return memo[i][j]

            if i == m - 1 and j == n - 1:
                return 1
            
            right_paths = dfs(i, j + 1) 
            down_paths = dfs(i + 1, j)   

            memo[i][j] = right_paths + down_paths
            return memo[i][j]
        
        memo = [[-1 for _ in range(n)] for _ in range(m)]
        return dfs(0, 0)

# Q63.) Unique Paths II

# You are given an m x n integer array grid. There is a robot initially located at the top-left corner (i.e., grid[0][0]). The robot tries to move to the bottom-right corner (i.e., grid[m - 1][n - 1]). The robot can only move either down or right at any point in time.

# An obstacle and space are marked as 1 or 0 respectively in grid. A path that the robot takes cannot include any square that is an obstacle.

# Return the number of possible unique paths that the robot can take to reach the bottom-right corner.

# The testcases are generated so that the answer will be less than or equal to 2 * 109.

# Example 1:

# Input: obstacleGrid = [[0,0,0],[0,1,0],[0,0,0]]
# Output: 2
# Explanation: There is one obstacle in the middle of the 3x3 grid above.
# There are two ways to reach the bottom-right corner:
# 1. Right -> Right -> Down -> Down
# 2. Down -> Down -> Right -> Right
# Example 2:


# Input: obstacleGrid = [[0,1],[0,0]]
# Output: 1
 
# Constraints:

# m == obstacleGrid.length
# n == obstacleGrid[i].length
# 1 <= m, n <= 100
# obstacleGrid[i][j] is 0 or 1.

# Sol_63}

class Solution:
    def uniquePathsWithObstacles(self, A: List[List[int]]) -> int:
        if A[0][0] or A[-1][-1]:
            return 0
        rangeN, rangeM, source = range(len(A)), range(len(A[0])), [(-1, 0), (0, -1)]
        A[0][0] = -1
        for i, j, (_i, _j) in product(rangeN, rangeM, source):
            if A[i][j] == 1:
                continue
            try:
                if i+_i != -1 and j+_j != -1 and A[i+_i][j+_j] != 1:
                    A[i][j] += A[i+_i][j+_j]
            except:
                pass
        return -A[-1][-1]
    
# Q64.) Minimum Path Sum

# Given a m x n grid filled with non-negative numbers, find a path from top left to bottom right, which minimizes the sum of all numbers along its path.

# Note: You can only move either down or right at any point in time.

# Example 1:

# Input: grid = [[1,3,1],[1,5,1],[4,2,1]]
# Output: 7
# Explanation: Because the path 1 → 3 → 1 → 1 → 1 minimizes the sum.
# Example 2:

# Input: grid = [[1,2,3],[4,5,6]]
# Output: 12

# Constraints:

# m == grid.length
# n == grid[i].length
# 1 <= m, n <= 200
# 0 <= grid[i][j] <= 200

# Sol_64}

class Solution:
    def minPathSum(self, grid: List[List[int]]) -> int:
        m, n = len(grid), len(grid[0])
        
        for i in range(1, n): 
            grid[0][i] += grid[0][i - 1]
        
        for i in range(1, m):  
            grid[i][0] += grid[i - 1][0]
        
        for i in range(1, m):
            for j in range(1, n):
                grid[i][j] += min(grid[i - 1][j], grid[i][j - 1])
        
        return grid[m - 1][n - 1]