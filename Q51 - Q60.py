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
    
# Q54.) Spiral Matrix

# Given an m x n matrix, return all elements of the matrix in spiral order.

# Example 1:

# Input: matrix = [[1,2,3],[4,5,6],[7,8,9]]
# Output: [1,2,3,6,9,8,7,4,5]
# Example 2:


# Input: matrix = [[1,2,3,4],[5,6,7,8],[9,10,11,12]]
# Output: [1,2,3,4,8,12,11,10,9,5,6,7]
 
# Constraints:

# m == matrix.length
# n == matrix[i].length
# 1 <= m, n <= 10
# -100 <= matrix[i][j] <= 100

# Sol_54}

class Solution:
    def spiralOrder(self, matrix: List[List[int]]) -> List[int]:
        if not matrix:
            return []
        
        result = []
        top, bottom, left, right = 0, len(matrix) - 1, 0, len(matrix[0]) - 1
        
        while top <= bottom and left <= right:
            # Traverse from left to right across the top row
            for i in range(left, right + 1):
                result.append(matrix[top][i])
            top += 1  # Move the top boundary down
            
            # Traverse from top to bottom down the right column
            for i in range(top, bottom + 1):
                result.append(matrix[i][right])
            right -= 1  # Move the right boundary left
            
            if top <= bottom:
                # Traverse from right to left across the bottom row
                for i in range(right, left - 1, -1):
                    result.append(matrix[bottom][i])
                bottom -= 1  # Move the bottom boundary up
            
            if left <= right:
                # Traverse from bottom to top up the left column
                for i in range(bottom, top - 1, -1):
                    result.append(matrix[i][left])
                left += 1  # Move the left boundary right
        
        return result

# Q55.) Jump Game

# You are given an integer array nums. You are initially positioned at the array's first index, and each element in the array represents your maximum jump length at that position.

# Return true if you can reach the last index, or false otherwise.

# Example 1:

# Input: nums = [2,3,1,1,4]
# Output: true
# Explanation: Jump 1 step from index 0 to 1, then 3 steps to the last index.
# Example 2:

# Input: nums = [3,2,1,0,4]
# Output: false
# Explanation: You will always arrive at index 3 no matter what. Its maximum jump length is 0, which makes it impossible to reach the last index.
 
# Constraints:

# 1 <= nums.length <= 104
# 0 <= nums[i] <= 105

# Sol_55}

class Solution:
    def canJump(self, nums: List[int]) -> bool:
        gas = 0
        for n in nums:
            if gas < 0:
                return False
            elif n > gas:
                gas = n
            gas -= 1
            
        return True        
    
# Q56.) Merge Intervals

# Given an array of intervals where intervals[i] = [starti, endi], merge all overlapping intervals, and return an array of the non-overlapping intervals that cover all the intervals in the input.

# Example 1:

# Input: intervals = [[1,3],[2,6],[8,10],[15,18]]
# Output: [[1,6],[8,10],[15,18]]
# Explanation: Since intervals [1,3] and [2,6] overlap, merge them into [1,6].
# Example 2:

# Input: intervals = [[1,4],[4,5]]
# Output: [[1,5]]
# Explanation: Intervals [1,4] and [4,5] are considered overlapping.
 
# Constraints:

# 1 <= intervals.length <= 104
# intervals[i].length == 2
# 0 <= starti <= endi <= 104

# Sol_56}

class Solution:
    def merge(self, intervals: List[List[int]]) -> List[List[int]]:
        merged = []
        intervals.sort(key=lambda x: x[0])

        prev = intervals[0]

        for interval in intervals[1:]:
            if interval[0] <= prev[1]:
                prev[1] = max(prev[1], interval[1])
            else:
                merged.append(prev)
                prev = interval
        
        merged.append(prev)

        return merged      


# Q57.) Insert Interval

# You are given an array of non-overlapping intervals intervals where intervals[i] = [starti, endi] represent the start and the end of the ith interval and intervals is sorted in ascending order by starti. You are also given an interval newInterval = [start, end] that represents the start and end of another interval.

# Insert newInterval into intervals such that intervals is still sorted in ascending order by starti and intervals still does not have any overlapping intervals (merge overlapping intervals if necessary).

# Return intervals after the insertion.

# Note that you don't need to modify intervals in-place. You can make a new array and return it.

# Example 1:

# Input: intervals = [[1,3],[6,9]], newInterval = [2,5]
# Output: [[1,5],[6,9]]
# Example 2:

# Input: intervals = [[1,2],[3,5],[6,7],[8,10],[12,16]], newInterval = [4,8]
# Output: [[1,2],[3,10],[12,16]]
# Explanation: Because the new interval [4,8] overlaps with [3,5],[6,7],[8,10].
 
# Constraints:

# 0 <= intervals.length <= 104
# intervals[i].length == 2
# 0 <= starti <= endi <= 105
# intervals is sorted by starti in ascending order.
# newInterval.length == 2
# 0 <= start <= end <= 105  
    
# Sol_57}

class Solution:
    def insert(self, intervals: List[List[int]], newInterval: List[int]) -> List[List[int]]:
        result = []
        i = 0
        n = len(intervals)

        # Add all intervals that come before the new interval
        while i < n and intervals[i][1] < newInterval[0]:
            result.append(intervals[i])
            i += 1

        # Merge intervals that overlap with the new interval
        while i < n and intervals[i][0] <= newInterval[1]:
            newInterval[0] = min(newInterval[0], intervals[i][0])
            newInterval[1] = max(newInterval[1], intervals[i][1])
            i += 1

        # Add the merged interval
        result.append(newInterval)

        # Add all intervals that come after the new interval
        result.extend(intervals[i:])

        return result

# Q58.) Length of Last Word

# Given a string s consisting of words and spaces, return the length of the last word in the string.

# A word is a maximal 
# substring
#  consisting of non-space characters only.

# Example 1:

# Input: s = "Hello World"
# Output: 5
# Explanation: The last word is "World" with length 5.
# Example 2:

# Input: s = "   fly me   to   the moon  "
# Output: 4
# Explanation: The last word is "moon" with length 4.
# Example 3:

# Input: s = "luffy is still joyboy"
# Output: 6
# Explanation: The last word is "joyboy" with length 6.
 

# Constraints:

# 1 <= s.length <= 104
# s consists of only English letters and spaces ' '.
# There will be at least one word in s.

# Sol_58}

class Solution:
    def lengthOfLastWord(self, s: str) -> int:
        words = s.strip().split()
        
        if not words:
            return 0
        
        return len(words[-1])

# Q59.) Spiral Matrix II

# Given a positive integer n, generate an n x n matrix filled with elements from 1 to n2 in spiral order.

# Example 1:

# Input: n = 3
# Output: [[1,2,3],[8,9,4],[7,6,5]]
# Example 2:

# Input: n = 1
# Output: [[1]]
 
# Constraints:

# 1 <= n <= 20

# Sol_59}

class Solution:
    def generateMatrix(self, n: int) -> List[List[int]]:
        arr = [[0] * n for _ in range(n)]
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)] 
        x, y, direction = 0, 0, 0 
        val = 1
        
        for _ in range(n * n):
            arr[x][y] = val
            val += 1
            next_x, next_y = x + directions[direction][0], y + directions[direction][1]
            if not (0 <= next_x < n and 0 <= next_y < n and arr[next_x][next_y] == 0):
                direction = (direction + 1) % 4
                next_x, next_y = x + directions[direction][0], y + directions[direction][1]
            x, y = next_x, next_y
        
        return arr
    
# Q60.) Permutation Sequence

# The set [1, 2, 3, ..., n] contains a total of n! unique permutations.

# By listing and labeling all of the permutations in order, we get the following sequence for n = 3:

# "123"
# "132"
# "213"
# "231"
# "312"
# "321"
# Given n and k, return the kth permutation sequence.

# Example 1:

# Input: n = 3, k = 3
# Output: "213"
# Example 2:

# Input: n = 4, k = 9
# Output: "2314"
# Example 3:

# Input: n = 3, k = 1
# Output: "123"
 
# Constraints:

# 1 <= n <= 9
# 1 <= k <= n!

# Sol_60}

class Solution:
    def getPermutation(self, n: int, k: int) -> str:
        numbers = list(range(1, n + 1))
        permutation = []  
        k -= 1 

        fact = 1
        for i in range(2, n):
            fact *= i  

        for i in range(n - 1, -1, -1):
            idx = k // fact
            permutation.append(str(numbers[idx])) 
            numbers.pop(idx)  

            if i > 0:
                k %= fact  
                fact //= i  

        return ''.join(permutation)