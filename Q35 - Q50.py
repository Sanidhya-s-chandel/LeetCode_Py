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
    
# Q42.) Trapping Rain Water

# Given n non-negative integers representing an elevation map where the width of each bar is 1, compute how much water it can trap after raining.

# Example 1:

# Input: height = [0,1,0,2,1,0,1,3,2,1,2,1]
# Output: 6
# Explanation: The above elevation map (black section) is represented by array [0,1,0,2,1,0,1,3,2,1,2,1]. In this case, 6 units of rain water (blue section) are being trapped.
# Example 2:

# Input: height = [4,2,0,3,2,5]
# Output: 9
 
# Constraints:

# n == height.length
# 1 <= n <= 2 * 104
# 0 <= height[i] <= 105

# Sol_42}

class Solution:
    def trap(self, height: List[int]) -> int:
        left = 0
        right = len(height) - 1
        left_max = height[left]
        right_max = height[right]
        water = 0

        while left < right:
            if left_max < right_max:
                left += 1
                left_max = max(left_max, height[left])
                water += left_max - height[left]
            else:
                right -= 1
                right_max = max(right_max, height[right])
                water += right_max - height[right]
        
        return water        

# Q43.) Multiply Strings

# Given two non-negative integers num1 and num2 represented as strings, return the product of num1 and num2, also represented as a string.

# Note: You must not use any built-in BigInteger library or convert the inputs to integer directly.

# Example 1:

# Input: num1 = "2", num2 = "3"
# Output: "6"
# Example 2:

# Input: num1 = "123", num2 = "456"
# Output: "56088"
 
# Constraints:

# 1 <= num1.length, num2.length <= 200
# num1 and num2 consist of digits only.
# Both num1 and num2 do not contain any leading zero, except the number 0 itself.

# Sol_43}

class Solution:
    def multiply(self, num1: str, num2: str) -> str:
        if num1 == '0' or num2 == '0':
            return '0'
        
        len1, len2 = len(num1), len(num2)
        result = [0] * (len1 + len2)

        for i in range(len1 - 1, -1, -1):
            for j in range(len2 - 1, -1, -1):
                mul = (ord(num1[i]) - ord('0')) * (ord(num2[j]) - ord('0'))
                pos1 = i + j
                pos2 = i + j + 1
                total = mul + result[pos2]

                result[pos2] = total % 10
                result[pos1] += total // 10

        result_str = ''.join(map(str, result)).lstrip('0')
        return result_str or '0'
    
# Q44.) Wildcard Matching

# Given an input string (s) and a pattern (p), implement wildcard pattern matching with support for '?' and '*' where:

# '?' Matches any single character.
# '*' Matches any sequence of characters (including the empty sequence).
# The matching should cover the entire input string (not partial).

# Example 1:

# Input: s = "aa", p = "a"
# Output: false
# Explanation: "a" does not match the entire string "aa".
# Example 2:

# Input: s = "aa", p = "*"
# Output: true
# Explanation: '*' matches any sequence.
# Example 3:

# Input: s = "cb", p = "?a"
# Output: false
# Explanation: '?' matches 'c', but the second letter is 'a', which does not match 'b'.
 
# Constraints:

# 0 <= s.length, p.length <= 2000
# s contains only lowercase English letters.
# p contains only lowercase English letters, '?' or '*'.

# Sol_44}

class Solution(object):
    def isMatch(self, string, pattern):
        first, second = 0, 0
        length1, length2 = len(pattern), len(string)
        star_index, match_index = -1, -1

        while second < length2:
            if first < length1 and (pattern[first] == string[second] or pattern[first] == '?'):
                first += 1
                second += 1
            elif first < length1 and pattern[first] == '*':
                star_index = first
                match_index = second
                first += 1
            elif star_index != -1:
                first = star_index + 1
                match_index += 1
                second = match_index
            else:
                return 0

        while first < length1 and pattern[first] == '*':
            first += 1

        return int(first == length1)

# Q45.) Jump Game II

# You are given a 0-indexed array of integers nums of length n. You are initially positioned at nums[0].

# Each element nums[i] represents the maximum length of a forward jump from index i. In other words, if you are at nums[i], you can jump to any nums[i + j] where:

# 0 <= j <= nums[i] and
# i + j < n
# Return the minimum number of jumps to reach nums[n - 1]. The test cases are generated such that you can reach nums[n - 1].

# Example 1:

# Input: nums = [2,3,1,1,4]
# Output: 2
# Explanation: The minimum number of jumps to reach the last index is 2. Jump 1 step from index 0 to 1, then 3 steps to the last index.
# Example 2:

# Input: nums = [2,3,0,1,4]
# Output: 2
 
# Constraints:

# 1 <= nums.length <= 104
# 0 <= nums[i] <= 1000
# It's guaranteed that you can reach nums[n - 1].

# Sol_45}

class Solution:
    def jump(self, nums: List[int]) -> int:
        jumps = 0
        current_end = 0
        farthest = 0
        
        for i in range(len(nums) - 1):
            farthest = max(farthest, i + nums[i])
            
            if i == current_end:
                jumps += 1
                current_end = farthest

        return jumps
    
# Q46.) Permutations

# Given an array nums of distinct integers, return all the possible 
# permutations
# . You can return the answer in any order.

# Example 1:

# Input: nums = [1,2,3]
# Output: [[1,2,3],[1,3,2],[2,1,3],[2,3,1],[3,1,2],[3,2,1]]
# Example 2:

# Input: nums = [0,1]
# Output: [[0,1],[1,0]]
# Example 3:

# Input: nums = [1]
# Output: [[1]]
 
# Constraints:

# 1 <= nums.length <= 6
# -10 <= nums[i] <= 10
# All the integers of nums are unique.

# Sol_46}

class Solution:
    def permute(self, nums: List[int]) -> List[List[int]]:
        def heap_permute(n, nums):
            if n == 1:
                result.append(nums[:])
                return
            
            for i in range(n):
                heap_permute(n - 1, nums)
                if n % 2 == 0:
                    nums[i], nums[n - 1] = nums[n - 1], nums[i] 
                else:
                    nums[0], nums[n - 1] = nums[n - 1], nums[0]  

        result = []
        heap_permute(len(nums), nums)
        return result

# Q47.) Permutations II

# Given a collection of numbers, nums, that might contain duplicates, return all possible unique permutations in any order.

# Example 1:

# Input: nums = [1,1,2]
# Output:
# [[1,1,2],
#  [1,2,1],
#  [2,1,1]]
# Example 2:

# Input: nums = [1,2,3]
# Output: [[1,2,3],[1,3,2],[2,1,3],[2,3,1],[3,1,2],[3,2,1]]
 
# Constraints:

# 1 <= nums.length <= 8
# -10 <= nums[i] <= 10

# Sol_47}

class Solution:
    def permuteUnique(self, nums: List[int]) -> List[List[int]]:
        def backtrack(path, used):
            if len(path) == len(nums):
                res.append(path[:])
                return
            
            for i in range(len(nums)):
                if used[i] or (i > 0 and nums[i] == nums[i - 1] and not used[i - 1]):
                    continue
                
                used[i] = True
                path.append(nums[i])
                
                backtrack(path, used)
                
                used[i] = False
                path.pop()
        
        nums.sort()  
        res = []
        backtrack([], [False] * len(nums))
        return res
    
# Q48.) Rotate Image

# You are given an n x n 2D matrix representing an image, rotate the image by 90 degrees (clockwise).

# You have to rotate the image in-place, which means you have to modify the input 2D matrix directly. DO NOT allocate another 2D matrix and do the rotation.

# Example 1:

# Input: matrix = [[1,2,3],[4,5,6],[7,8,9]]
# Output: [[7,4,1],[8,5,2],[9,6,3]]
# Example 2:

# Input: matrix = [[5,1,9,11],[2,4,8,10],[13,3,6,7],[15,14,12,16]]
# Output: [[15,13,2,5],[14,3,4,1],[12,6,8,9],[16,7,10,11]]
 
# Constraints:

# n == matrix.length == matrix[i].length
# 1 <= n <= 20
# -1000 <= matrix[i][j] <= 1000

# Sol_48}

class Solution:
    def rotate(self, matrix: List[List[int]]) -> None:
        edge_length = len(matrix)

        top = 0
        bottom = edge_length - 1

        while top < bottom:
            for col in range(edge_length):
                temp = matrix[top][col]
                matrix[top][col] = matrix[bottom][col]
                matrix[bottom][col] = temp
            top += 1
            bottom -= 1

        for row in range(edge_length):
            for col in range(row+1, edge_length):
                temp = matrix[row][col]
                matrix[row][col] = matrix[col][row]
                matrix[col][row] = temp
        
        return matrix

# Q49.) Group Anagrams

# Given an array of strings strs, group the 
# anagrams together. You can return the answer in any order.

# Example 1:

# Input: strs = ["eat","tea","tan","ate","nat","bat"]

# Output: [["bat"],["nat","tan"],["ate","eat","tea"]]

# Explanation:

# There is no string in strs that can be rearranged to form "bat".
# The strings "nat" and "tan" are anagrams as they can be rearranged to form each other.
# The strings "ate", "eat", and "tea" are anagrams as they can be rearranged to form each other.
# Example 2:

# Input: strs = [""]

# Output: [[""]]

# Example 3:

# Input: strs = ["a"]

# Output: [["a"]]

# Constraints:

# 1 <= strs.length <= 104
# 0 <= strs[i].length <= 100
# strs[i] consists of lowercase English letters.

# Sol_49}

class Solution:
    def groupAnagrams(self, strs: List[str]) -> List[List[str]]:
        ans = defaultdict(list)

        for s in strs:
            key = "".join(sorted(s))
            ans[key].append(s)
        
        return ans.values()        
    
# Q50.) Pow(x, n)

# Implement pow(x, n), which calculates x raised to the power n (i.e., xn).

# Example 1:

# Input: x = 2.00000, n = 10
# Output: 1024.00000
# Example 2:

# Input: x = 2.10000, n = 3
# Output: 9.26100
# Example 3:

# Input: x = 2.00000, n = -2
# Output: 0.25000
# Explanation: 2-2 = 1/22 = 1/4 = 0.25
 
# Constraints:

# -100.0 < x < 100.0
# -231 <= n <= 231-1
# n is an integer.
# Either x is not zero or n > 0.
# -104 <= xn <= 104

# Sol_50}

class Solution:
    def myPow(self, x: float, n: int) -> float:
        def calc_power(x, n):
            if x == 0:
                return 0
            if n == 0:
                return 1
            
            res = calc_power(x, n // 2)
            res = res * res

            if n % 2 == 1:
                return res * x
            
            return res

        ans = calc_power(x, abs(n))

        if n >= 0:
            return ans
        
        return 1 / ans         