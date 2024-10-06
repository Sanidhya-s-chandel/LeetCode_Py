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

# Q65.) Valid Number

# Given a string s, return whether s is a valid number.

# For example, all the following are valid numbers: "2", "0089", "-0.1", "+3.14", "4.", "-.9", "2e10", "-90E3", "3e+7", "+6e-1", "53.5e93", "-123.456e789", while the following are not valid numbers: "abc", "1a", "1e", "e3", "99e2.5", "--6", "-+3", "95a54e53".

# Formally, a valid number is defined using one of the following definitions:

# An integer number followed by an optional exponent.
# A decimal number followed by an optional exponent.
# An integer number is defined with an optional sign '-' or '+' followed by digits.

# A decimal number is defined with an optional sign '-' or '+' followed by one of the following definitions:

# Digits followed by a dot '.'.
# Digits followed by a dot '.' followed by digits.
# A dot '.' followed by digits.
# An exponent is defined with an exponent notation 'e' or 'E' followed by an integer number.

# The digits are defined as one or more digits.

# Example 1:

# Input: s = "0"

# Output: true

# Example 2:

# Input: s = "e"

# Output: false

# Example 3:

# Input: s = "."

# Output: false


# Constraints:

# 1 <= s.length <= 20
# s consists of only English letters (both uppercase and lowercase), digits (0-9), plus '+', minus '-', or dot '.'.

# Sol_65}

class Solution:
    def isNumber(self, S: str) -> bool:    
        num, exp, sign, dec = False, False, False, False
        for c in S:
            if c >= '0' and c <= '9': num = True     
            elif c == 'e' or c == 'E':
                if exp or not num: return False
                else: exp, num, sign, dec = True, False, False, False
            elif c == '+' or c == '-':
                if sign or num or dec: return False
                else: sign = True
            elif c == '.':
                if dec or exp: return False
                else: dec = True
            else: return False
        return num

# Q66.) Plus One

# You are given a large integer represented as an integer array digits, where each digits[i] is the ith digit of the integer. The digits are ordered from most significant to least significant in left-to-right order. The large integer does not contain any leading 0's.

# Increment the large integer by one and return the resulting array of digits.

# Example 1:

# Input: digits = [1,2,3]
# Output: [1,2,4]
# Explanation: The array represents the integer 123.
# Incrementing by one gives 123 + 1 = 124.
# Thus, the result should be [1,2,4].
# Example 2:

# Input: digits = [4,3,2,1]
# Output: [4,3,2,2]
# Explanation: The array represents the integer 4321.
# Incrementing by one gives 4321 + 1 = 4322.
# Thus, the result should be [4,3,2,2].
# Example 3:

# Input: digits = [9]
# Output: [1,0]
# Explanation: The array represents the integer 9.
# Incrementing by one gives 9 + 1 = 10.
# Thus, the result should be [1,0].
 
# Constraints:

# 1 <= digits.length <= 100
# 0 <= digits[i] <= 9
# digits does not contain any leading 0's.

# Sol_66}

class Solution:
    def plusOne(self, digits: List[int]) -> List[int]:
        for i in reversed(range(len(digits))):
            if digits[i] != 9:
                digits[i] += 1
                return digits
            digits[i] = 0
        
        return [1] + digits

# Q67.) Add Binary

# Given two binary strings a and b, return their sum as a binary string.

# Example 1:

# Input: a = "11", b = "1"
# Output: "100"
# Example 2:

# Input: a = "1010", b = "1011"
# Output: "10101"
 
# Constraints:

# 1 <= a.length, b.length <= 104
# a and b consist only of '0' or '1' characters.
# Each string does not contain leading zeros except for the zero itself.

# Sol_67}

class Solution:
  def addBinary(self, a: str, b: str) -> str:
    s = []
    carry = 0
    i = len(a) - 1
    j = len(b) - 1

    while i >= 0 or j >= 0 or carry:
      if i >= 0:
        carry += int(a[i])
        i -= 1
      if j >= 0:
        carry += int(b[j])
        j -= 1
      s.append(str(carry % 2))
      carry //= 2

    return ''.join(reversed(s))

# Q68.) Text Justification

# Given an array of strings words and a width maxWidth, format the text such that each line has exactly maxWidth characters and is fully (left and right) justified.

# You should pack your words in a greedy approach; that is, pack as many words as you can in each line. Pad extra spaces ' ' when necessary so that each line has exactly maxWidth characters.

# Extra spaces between words should be distributed as evenly as possible. If the number of spaces on a line does not divide evenly between words, the empty slots on the left will be assigned more spaces than the slots on the right.

# For the last line of text, it should be left-justified, and no extra space is inserted between words.

# Note:

# A word is defined as a character sequence consisting of non-space characters only.
# Each word's length is guaranteed to be greater than 0 and not exceed maxWidth.
# The input array words contains at least one word.
 

# Example 1:

# Input: words = ["This", "is", "an", "example", "of", "text", "justification."], maxWidth = 16
# Output:
# [
#    "This    is    an",
#    "example  of text",
#    "justification.  "
# ]
# Example 2:

# Input: words = ["What","must","be","acknowledgment","shall","be"], maxWidth = 16
# Output:
# [
#   "What   must   be",
#   "acknowledgment  ",
#   "shall be        "
# ]
# Explanation: Note that the last line is "shall be    " instead of "shall     be", because the last line must be left-justified instead of fully-justified.
# Note that the second line is also left-justified because it contains only one word.
# Example 3:

# Input: words = ["Science","is","what","we","understand","well","enough","to","explain","to","a","computer.","Art","is","everything","else","we","do"], maxWidth = 20
# Output:
# [
#   "Science  is  what we",
#   "understand      well",
#   "enough to explain to",
#   "a  computer.  Art is",
#   "everything  else  we",
#   "do                  "
# ]
 
# Constraints:

# 1 <= words.length <= 300
# 1 <= words[i].length <= 20
# words[i] consists of only English letters and symbols.
# 1 <= maxWidth <= 100
# words[i].length <= maxWidth

# Sol_68}

class Solution:
    def fullJustify(self, words, maxWidth):
        res, cur, num_of_letters = [], [], 0
        for w in words:
            if num_of_letters + len(w) + len(cur) > maxWidth:
                for i in range(maxWidth - num_of_letters):
                    cur[i%(len(cur)-1 or 1)] += ' '
                res.append(''.join(cur))
                cur, num_of_letters = [], 0
            cur += [w]
            num_of_letters += len(w)
        return res + [' '.join(cur).ljust(maxWidth)] 

# Q69.) Sqrt(x)

# Given a non-negative integer x, return the square root of x rounded down to the nearest integer. The returned integer should be non-negative as well.

# You must not use any built-in exponent function or operator.

# For example, do not use pow(x, 0.5) in c++ or x ** 0.5 in python.
 
# Example 1:

# Input: x = 4
# Output: 2
# Explanation: The square root of 4 is 2, so we return 2.
# Example 2:

# Input: x = 8
# Output: 2
# Explanation: The square root of 8 is 2.82842..., and since we round it down to the nearest integer, 2 is returned.
 
# Constraints:

# 0 <= x <= 231 - 1

# Sol_69}

class Solution:
    def mySqrt(self, x: int) -> int:
        if x == 0:
            return 0
        
        left, right = 1, x
        result = 0
        
        while left <= right:
            mid = (left + right) // 2
            mid_squared = mid * mid
            
            if mid_squared == x:
                return mid
            elif mid_squared < x:
                result = mid
                left = mid + 1
            else:
                right = mid - 1
        
        return result

# 70.) Climbing Stairs

# You are climbing a staircase. It takes n steps to reach the top.

# Each time you can either climb 1 or 2 steps. In how many distinct ways can you climb to the top?

# Example 1:

# Input: n = 2
# Output: 2
# Explanation: There are two ways to climb to the top.
# 1. 1 step + 1 step
# 2. 2 steps
# Example 2:

# Input: n = 3
# Output: 3
# Explanation: There are three ways to climb to the top.
# 1. 1 step + 1 step + 1 step
# 2. 1 step + 2 steps
# 3. 2 steps + 1 step
 
# Constraints:

# 1 <= n <= 45

# Sol_70}

class Solution:
    def climbStairs(self, n: int) -> int:
        memo = {}
        return self.helper(n, memo)
    
    def helper(self, n: int, memo: dict[int, int]) -> int:
        if n == 0 or n == 1:
            return 1
        if n not in memo:
            memo[n] = self.helper(n-1, memo) + self.helper(n-2, memo)
        return memo[n]