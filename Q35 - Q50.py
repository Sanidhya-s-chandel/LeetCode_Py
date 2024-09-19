# Q36.) Valid Sudoku

# Determine if a 9 x 9 Sudoku board is valid. Only the filled cells need to be validated according to the following rules:

# Each row must contain the digits 1-9 without repetition.
# Each column must contain the digits 1-9 without repetition.
# Each of the nine 3 x 3 sub-boxes of the grid must contain the digits 1-9 without repetition.
# Note:

# A Sudoku board (partially filled) could be valid but is not necessarily solvable.
# Only the filled cells need to be validated according to the mentioned rules.
 
# Example 1:


# Input: board = 
# [["5","3",".",".","7",".",".",".","."]
# ,["6",".",".","1","9","5",".",".","."]
# ,[".","9","8",".",".",".",".","6","."]
# ,["8",".",".",".","6",".",".",".","3"]
# ,["4",".",".","8",".","3",".",".","1"]
# ,["7",".",".",".","2",".",".",".","6"]
# ,[".","6",".",".",".",".","2","8","."]
# ,[".",".",".","4","1","9",".",".","5"]
# ,[".",".",".",".","8",".",".","7","9"]]
# Output: true
# Example 2:

# Input: board = 
# [["8","3",".",".","7",".",".",".","."]
# ,["6",".",".","1","9","5",".",".","."]
# ,[".","9","8",".",".",".",".","6","."]
# ,["8",".",".",".","6",".",".",".","3"]
# ,["4",".",".","8",".","3",".",".","1"]
# ,["7",".",".",".","2",".",".",".","6"]
# ,[".","6",".",".",".",".","2","8","."]
# ,[".",".",".","4","1","9",".",".","5"]
# ,[".",".",".",".","8",".",".","7","9"]]
# Output: false
# Explanation: Same as Example 1, except with the 5 in the top left corner being modified to 8. Since there are two 8's in the top left 3x3 sub-box, it is invalid.
 
# Constraints:
# board.length == 9
# board[i].length == 9
# board[i][j] is a digit 1-9 or '.'.

# Sol_36}

class Solution:
    def isValidSudoku(self, board: List[List[str]]) -> bool:
        rows = [0] * 9
        cols = [0] * 9
        boxes = [0] * 9
        
        for r in range(9):
            for c in range(9):
                if board[r][c] == ".":
                    continue
                
                num = int(board[r][c])
                pos = 1 << (num - 1)
                
                if rows[r] & pos or cols[c] & pos or boxes[(r // 3) * 3 + c // 3] & pos:
                    return False
                
                rows[r] |= pos
                cols[c] |= pos
                boxes[(r // 3) * 3 + c // 3] |= pos
        
        return True
    
# Q37.) Sudoku Solver

# Write a program to solve a Sudoku puzzle by filling the empty cells.

# A sudoku solution must satisfy all of the following rules:

# Each of the digits 1-9 must occur exactly once in each row.
# Each of the digits 1-9 must occur exactly once in each column.
# Each of the digits 1-9 must occur exactly once in each of the 9 3x3 sub-boxes of the grid.
# The '.' character indicates empty cells.

# Example 1:

# Input: board = [["5","3",".",".","7",".",".",".","."],["6",".",".","1","9","5",".",".","."],[".","9","8",".",".",".",".","6","."],["8",".",".",".","6",".",".",".","3"],["4",".",".","8",".","3",".",".","1"],["7",".",".",".","2",".",".",".","6"],[".","6",".",".",".",".","2","8","."],[".",".",".","4","1","9",".",".","5"],[".",".",".",".","8",".",".","7","9"]]
# Output: [["5","3","4","6","7","8","9","1","2"],["6","7","2","1","9","5","3","4","8"],["1","9","8","3","4","2","5","6","7"],["8","5","9","7","6","1","4","2","3"],["4","2","6","8","5","3","7","9","1"],["7","1","3","9","2","4","8","5","6"],["9","6","1","5","3","7","2","8","4"],["2","8","7","4","1","9","6","3","5"],["3","4","5","2","8","6","1","7","9"]]
# Explanation: The input board is shown above and the only valid solution is shown below:

# Constraints:

# board.length == 9
# board[i].length == 9
# board[i][j] is a digit or '.'.
# It is guaranteed that the input board has only one solution.

# Sol_37}

class Solution:
    def solveSudoku(self, board: List[List[str]]) -> None:
        def canPlace(d, row, col):
            return not (d in rows[row] or d in cols[col] or d in boxes[(row // 3) * 3 + col // 3])

        def placeNumber(d, row, col):
            rows[row].add(d)
            cols[col].add(d)
            boxes[(row // 3) * 3 + col // 3].add(d)
            board[row][col] = d

        def removeNumber(d, row, col):
            rows[row].remove(d)
            cols[col].remove(d)
            boxes[(row // 3) * 3 + col // 3].remove(d)
            board[row][col] = '.'

        def placeNextNumbers(row, col):
            if row == N - 1 and col == N - 1:
                nonlocal sudoku_solved
                sudoku_solved = True
            elif col == N - 1:
                backtrack(row + 1, 0)
            else:
                backtrack(row, col + 1)

        def backtrack(row=0, col=0):
            if board[row][col] == '.':
                for d in range(1, 10):
                    D = str(d)
                    if canPlace(D, row, col):
                        placeNumber(D, row, col)
                        placeNextNumbers(row, col)
                        if not sudoku_solved:
                            removeNumber(D, row, col)
            else:
                placeNextNumbers(row, col)

        n = 3
        N = n * n
        rows = [set() for _ in range(N)]
        cols = [set() for _ in range(N)]
        boxes = [set() for _ in range(N)]
        sudoku_solved = False

        for i in range(N):
            for j in range(N):
                if board[i][j] != '.':
                    D = board[i][j]
                    rows[i].add(D)
                    cols[j].add(D)
                    boxes[(i // 3) * 3 + j // 3].add(D)

        backtrack()

# Q38.) Count and Say

# The count-and-say sequence is a sequence of digit strings defined by the recursive formula:

# countAndSay(1) = "1"
# countAndSay(n) is the run-length encoding of countAndSay(n - 1).
# Run-length encoding (RLE) is a string compression method that works by replacing consecutive identical characters (repeated 2 or more times) with the concatenation of the character and the number marking the count of the characters (length of the run). For example, to compress the string "3322251" we replace "33" with "23", replace "222" with "32", replace "5" with "15" and replace "1" with "11". Thus the compressed string becomes "23321511".

# Given a positive integer n, return the nth element of the count-and-say sequence.

# Example 1:

# Input: n = 4

# Output: "1211"

# Explanation:

# countAndSay(1) = "1"
# countAndSay(2) = RLE of "1" = "11"
# countAndSay(3) = RLE of "11" = "21"
# countAndSay(4) = RLE of "21" = "1211"
# Example 2:

# Input: n = 1

# Output: "1"

# Explanation:

# This is the base case.

# Constraints:

# 1 <= n <= 30

# Sol_38}

class Solution:
    def countAndSay(self, n: int) -> str:
        def count(string):
            new_string = ''
            c = 1
            previous = string[0]
            for i in range(1, len(string)):
                if string[i] == previous:
                    c += 1
                else:
                    new_string += str(c) + previous
                    c = 1
                    previous = string[i]
            new_string += str(c) + previous
            return new_string
        
        result = '1'
        for i in range(1, n):
            result = count(result)
        
        return result
    
# Q39.) Combination Sum

# Given an array of distinct integers candidates and a target integer target, return a list of all unique combinations of candidates where the chosen numbers sum to target. You may return the combinations in any order.

# The same number may be chosen from candidates an unlimited number of times. Two combinations are unique if the 
# frequency
#  of at least one of the chosen numbers is different.

# The test cases are generated such that the number of unique combinations that sum up to target is less than 150 combinations for the given input.

# Example 1:

# Input: candidates = [2,3,6,7], target = 7
# Output: [[2,2,3],[7]]
# Explanation:
# 2 and 3 are candidates, and 2 + 2 + 3 = 7. Note that 2 can be used multiple times.
# 7 is a candidate, and 7 = 7.
# These are the only two combinations.
# Example 2:

# Input: candidates = [2,3,5], target = 8
# Output: [[2,2,2,2],[2,3,3],[3,5]]
# Example 3:

# Input: candidates = [2], target = 1
# Output: []
 
# Constraints:

# 1 <= candidates.length <= 30
# 2 <= candidates[i] <= 40
# All elements of candidates are distinct.
# 1 <= target <= 40

# Sol_39}

class Solution:
    def combinationSum(self, candidates: List[int], target: int) -> List[List[int]]:
        dp = [[] for _ in range(target + 1)]
        dp[0] = [[]] 
        
        for candidate in candidates:
            for t in range(candidate, target + 1):
                for comb in dp[t - candidate]:
                    dp[t].append(comb + [candidate])
        
        return dp[target]

# Q40.) Combination Sum II

# Given a collection of candidate numbers (candidates) and a target number (target), find all unique combinations in candidates where the candidate numbers sum to target.

# Each number in candidates may only be used once in the combination.

# Note: The solution set must not contain duplicate combinations.

# Example 1:

# Input: candidates = [10,1,2,7,6,1,5], target = 8
# Output: 
# [
# [1,1,6],
# [1,2,5],
# [1,7],
# [2,6]
# ]
# Example 2:

# Input: candidates = [2,5,2,1,2], target = 5
# Output: 
# [
# [1,2,2],
# [5]
# ]
 
# Constraints:

# 1 <= candidates.length <= 100
# 1 <= candidates[i] <= 50
# 1 <= target <= 30

# Sol_40}

class Solution:
    def combinationSum2(self, candidates: List[int], target: int) -> List[List[int]]:
        candidates.sort()
        res = []

        def dfs(target, start, comb):
            if target == 0:
                res.append(comb)
                return
            for i in range(start, len(candidates)):
                if i > start and candidates[i] == candidates[i - 1]:
                    continue
                if candidates[i] > target:
                    break  # Prune the branch
                dfs(target - candidates[i], i + 1, comb + [candidates[i]])

        dfs(target, 0, [])
        return res

# Q41.) First Missing Positive

# Given an unsorted integer array nums. Return the smallest positive integer that is not present in nums.

# You must implement an algorithm that runs in O(n) time and uses O(1) auxiliary space.

# Example 1:

# Input: nums = [1,2,0]
# Output: 3
# Explanation: The numbers in the range [1,2] are all in the array.
# Example 2:

# Input: nums = [3,4,-1,1]
# Output: 2
# Explanation: 1 is in the array but 2 is missing.
# Example 3:

# Input: nums = [7,8,9,11,12]
# Output: 1
# Explanation: The smallest positive integer 1 is missing.
 
# Constraints:

# 1 <= nums.length <= 105
# -231 <= nums[i] <= 231 - 1

# Sol_41}

class Solution:
    def firstMissingPositive(self, nums: List[int]) -> int:
        num_set = set(num for num in nums if num > 0)
        
        i = 1
        while i in num_set:
            i += 1
        
        return i