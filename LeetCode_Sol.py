# Q1) Two Sum

# Given an array of integers nums and an integer target, return indices of the two numbers such that they add up to target.

# You may assume that each input would have exactly one solution, and you may not use the same element twice.

# You can return the answer in any order.

# Example 1:

# Input: nums = [2,7,11,15], target = 9
# Output: [0,1]
# Explanation: Because nums[0] + nums[1] == 9, we return [0, 1].
# Example 2:

# Input: nums = [3,2,4], target = 6
# Output: [1,2]
# Example 3:

# Input: nums = [3,3], target = 6
# Output: [0,1]

# sol_1}

# with Time Complexity O(n^2)

class Solution:
    def twoSum(self, nums: List[int], target: int) -> List[int]:
        
        seen = {}

        for i,j in enumerate(nums):
            if target - j in seen:
                return([seen[target - j],i])
            
            elif j not in seen:
                seen[j] = i

# Brute Force Method

class Solution:
    def twoSum(self, nums: List[int], target: int) -> List[int]:
        
        for _ in range(len(nums)-1):
            for i in range(_+1,len(nums)):
                if nums[_] + nums[i] == target:
                    return([_,i])


# Q2) ADD Two Numbers

# You are given two non-empty linked lists representing two non-negative integers. The digits are stored in reverse order, and each of their nodes contains a single digit. Add the two numbers and return the sum as a linked list.

# You may assume the two numbers do not contain any leading zero, except the number 0 itself.

# Example 1:


# Input: l1 = [2,4,3], l2 = [5,6,4]
# Output: [7,0,8]
# Explanation: 342 + 465 = 807.
# Example 2:

# Input: l1 = [0], l2 = [0]
# Output: [0]
# Example 3:

# Input: l1 = [9,9,9,9,9,9,9], l2 = [9,9,9,9]
# Output: [8,9,9,9,0,0,0,1]

# Sol_2}

# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next

class Solution:
    def addTwoNumbers(self, l1: Optional[ListNode], l2: Optional[ListNode]) -> Optional[ListNode]:
        dummy = ListNode()
        current = dummy
        carry = 0
        
        while l1 or l2 or carry:
            val1 = l1.val if l1 else 0
            val2 = l2.val if l2 else 0
            total = val1 + val2 + carry
            carry = total // 10
            current.next = ListNode(total % 10)
            current = current.next
            
            if l1:
                l1 = l1.next
            if l2:
                l2 = l2.next
        
        return dummy.next
    
# Q3) Longest Substring Without Repeating Characters

# Given a string s, find the length of the longest 
# substring
#  without repeating characters.


# Example 1:

# Input: s = "abcabcbb"
# Output: 3
# Explanation: The answer is "abc", with the length of 3.
# Example 2:

# Input: s = "bbbbb"
# Output: 1
# Explanation: The answer is "b", with the length of 1.
# Example 3:

# Input: s = "pwwkew"
# Output: 3
# Explanation: The answer is "wke", with the length of 3.
# Notice that the answer must be a substring, "pwke" is a subsequence and not a substring.

# sol_3}

class Solution:
    def lengthOfLongestSubstring(self, s: str) -> int:
        n = len(s)
        if n == 0:
            return 0

        char_index_map = {}
        max_length = 0
        start = 0

        for end in range(n):
            if s[end] in char_index_map:
                start = max(start, char_index_map[s[end]] + 1)

            char_index_map[s[end]] = end

            max_length = max(max_length, end - start + 1)

        return max_length
    

# Q4) Median of Two Sorted Arrays

# Given two sorted arrays nums1 and nums2 of size m and n respectively, return the median of the two sorted arrays.

# The overall run time complexity should be O(log (m+n)).


# Example 1:

# Input: nums1 = [1,3], nums2 = [2]
# Output: 2.00000
# Explanation: merged array = [1,2,3] and median is 2.
# Example 2:

# Input: nums1 = [1,2], nums2 = [3,4]
# Output: 2.50000
# Explanation: merged array = [1,2,3,4] and median is (2 + 3) / 2 = 2.5.
 

# Constraints:

# nums1.length == m
# nums2.length == n
# 0 <= m <= 1000
# 0 <= n <= 1000
# 1 <= m + n <= 2000
# -106 <= nums1[i], nums2[i] <= 106

# sol_4}

class Solution:
    def findMedianSortedArrays(self, nums1: List[int], nums2: List[int]) -> float:
        if len(nums1) > len(nums2):
            nums1, nums2 = nums2, nums1
        
        x, y = len(nums1), len(nums2)
        low, high = 0, x
        
        while low <= high:
            partitionX = (low + high) // 2
            partitionY = (x + y + 1) // 2 - partitionX
            
            maxLeftX = float('-inf') if partitionX == 0 else nums1[partitionX - 1]
            minRightX = float('inf') if partitionX == x else nums1[partitionX]
            
            maxLeftY = float('-inf') if partitionY == 0 else nums2[partitionY - 1]
            minRightY = float('inf') if partitionY == y else nums2[partitionY]
            
            if maxLeftX <= minRightY and maxLeftY <= minRightX:
                if (x + y) % 2 == 0:
                    return (max(maxLeftX, maxLeftY) + min(minRightX, minRightY)) / 2
                else:
                    return max(maxLeftX, maxLeftY)
            elif maxLeftX > minRightY:
                high = partitionX - 1
            else:
                low = partitionX + 1
        raise ValueError("Input arrays are not sorted")
    

# Q5.) Longest Palindromic Substring

# Given a string s, return the longest 
# palindromic
 
# substring
#  in s.

# Example 1:

# Input: s = "babad"
# Output: "bab"
# Explanation: "aba" is also a valid answer.
# Example 2:

# Input: s = "cbbd"
# Output: "bb"
 

# Constraints:

# 1 <= s.length <= 1000
# s consist of only digits and English letters.

# Sol_5}

class Solution:
    def longestPalindrome(self, s: str) -> str:
        if not s:
            return ""

        start, end = 0, 0

        for i in range(len(s)):
            len1 = self.expandAroundCenter(s, i, i)
            len2 = self.expandAroundCenter(s, i, i + 1)
            max_len = max(len1, len2)
            
            if max_len > (end - start):
                start = i - (max_len - 1) // 2
                end = i + max_len // 2

        return s[start:end + 1]

    def expandAroundCenter(self, s: str, left: int, right: int) -> int:
        while left >= 0 and right < len(s) and s[left] == s[right]:
            left -= 1
            right += 1
        return right - left - 1

# Example usage:
sol = Solution()
s = "babad"
print(sol.longestPalindrome(s))

# Q6.) Zigzag Conversion

# The string "PAYPALISHIRING" is written in a zigzag pattern on a given number of rows like this: (you may want to display this pattern in a fixed font for better legibility)

# P   A   H   N
# A P L S I I G
# Y   I   R
# And then read line by line: "PAHNAPLSIIGYIR"

# Write the code that will take a string and make this conversion given a number of rows:

# string convert(string s, int numRows);
 

# Example 1:

# Input: s = "PAYPALISHIRING", numRows = 3
# Output: "PAHNAPLSIIGYIR"
# Example 2:

# Input: s = "PAYPALISHIRING", numRows = 4
# Output: "PINALSIGYAHRPI"
# Explanation:
# P     I    N
# A   L S  I G
# Y A   H R
# P     I
# Example 3:

# Input: s = "A", numRows = 1
# Output: "A"
 

# Constraints:

# 1 <= s.length <= 1000
# s consists of English letters (lower-case and upper-case), ',' and '.'.
# 1 <= numRows <= 1000

# Sol_6}

class Solution:
    def convert(self, s: str, numRows: int) -> str:
        if numRows == 1 or numRows >= len(s):
            return s

        idx, d = 0, 1
        rows = [[] for _ in range(numRows)]

        for char in s:
            rows[idx].append(char)
            if idx == 0:
                d = 1
            elif idx == numRows - 1:
                d = -1
            idx += d

        for i in range(numRows):
            rows[i] = ''.join(rows[i])

        return ''.join(rows)
    

# Q7.) Reverse Integer

# Given a signed 32-bit integer x, return x with its digits reversed. If reversing x causes the value to go outside the signed 32-bit integer range [-231, 231 - 1], then return 0.

# Assume the environment does not allow you to store 64-bit integers (signed or unsigned).

# Example 1:

# Input: x = 123
# Output: 321
# Example 2:

# Input: x = -123
# Output: -321
# Example 3:

# Input: x = 120
# Output: 21
 

# Constraints:

# -231 <= x <= 231 - 1

# Sol_7}

class Solution:
    def reverse(self, x: int) -> int:
        reverse = 0
        is_negative = x < 0
        x = abs(x)

        while x != 0:
            reverse = reverse * 10 + x % 10
            x //= 10

        if is_negative:
            reverse = -reverse

        if reverse < -2**31 or reverse > 2**31 - 1:
            return 0

        return reverse
    

# Q8.) String to Integer (atoi)

# Implement the myAtoi(string s) function, which converts a string to a 32-bit signed integer.

# The algorithm for myAtoi(string s) is as follows:

# Whitespace: Ignore any leading whitespace (" ").
# Signedness: Determine the sign by checking if the next character is '-' or '+', assuming positivity is neither present.
# Conversion: Read the integer by skipping leading zeros until a non-digit character is encountered or the end of the string is reached. If no digits were read, then the result is 0.
# Rounding: If the integer is out of the 32-bit signed integer range [-231, 231 - 1], then round the integer to remain in the range. Specifically, integers less than -231 should be rounded to -231, and integers greater than 231 - 1 should be rounded to 231 - 1.
# Return the integer as the final result.

# Example 1:

# Input: s = "42"

# Output: 42

# Explanation:

# The underlined characters are what is read in and the caret is the current reader position.
# Step 1: "42" (no characters read because there is no leading whitespace)
#          ^
# Step 2: "42" (no characters read because there is neither a '-' nor '+')
#          ^
# Step 3: "42" ("42" is read in)
#            ^
# Example 2:

# Input: s = " -042"

# Output: -42

# Explanation:

# Step 1: "   -042" (leading whitespace is read and ignored)
#             ^
# Step 2: "   -042" ('-' is read, so the result should be negative)
#              ^
# Step 3: "   -042" ("042" is read in, leading zeros ignored in the result)
#                ^
# Example 3:

# Input: s = "1337c0d3"

# Output: 1337

# Explanation:

# Step 1: "1337c0d3" (no characters read because there is no leading whitespace)
#          ^
# Step 2: "1337c0d3" (no characters read because there is neither a '-' nor '+')
#          ^
# Step 3: "1337c0d3" ("1337" is read in; reading stops because the next character is a non-digit)
#              ^
# Example 4:

# Input: s = "0-1"

# Output: 0

# Explanation:

# Step 1: "0-1" (no characters read because there is no leading whitespace)
#          ^
# Step 2: "0-1" (no characters read because there is neither a '-' nor '+')
#          ^
# Step 3: "0-1" ("0" is read in; reading stops because the next character is a non-digit)
#           ^
# Example 5:

# Input: s = "words and 987"

# Output: 0

# Explanation:

# Reading stops at the first non-digit character 'w'.

# Constraints:

# 0 <= s.length <= 200
# s consists of English letters (lower-case and upper-case), digits (0-9), ' ', '+', '-', and '.'.

# Sol_8}

class Solution:
    def myAtoi(self, s: str) -> int:
        s = s.lstrip()

        if not s:
            return 0

        sign = 1
        index = 0

        if s[0] == '-':
            sign = -1
            index += 1
        elif s[0] == '+':
            index += 1

        num = 0
        while index < len(s) and s[index].isdigit():
            num = num * 10 + int(s[index])
            index += 1

        num *= sign

        INT_MIN = -2**31
        INT_MAX = 2**31 - 1

        if num < INT_MIN:
            return INT_MIN
        if num > INT_MAX:
            return INT_MAX
        
        return num
    

# Q9.) Palindrome Number

# Given an integer x, return true if x is a 
# palindrome
# , and false otherwise.

# Example 1:

# Input: x = 121
# Output: true
# Explanation: 121 reads as 121 from left to right and from right to left.
# Example 2:

# Input: x = -121
# Output: false
# Explanation: From left to right, it reads -121. From right to left, it becomes 121-. Therefore it is not a palindrome.
# Example 3:

# Input: x = 10
# Output: false
# Explanation: Reads 01 from right to left. Therefore it is not a palindrome.
 

# Constraints:

# -231 <= x <= 231 - 1

# Sol_9}

class Solution:
    def isPalindrome(self, x: int) -> bool:
        copy = x
        s = 0
        while x > 0:
            s = s * 10 + x % 10
            x //= 10
        return s == copy
    

# Q10.) Regular Expression Matching

# Given an input string s and a pattern p, implement regular expression matching with support for '.' and '*' where:

# '.' Matches any single character.​​​​
# '*' Matches zero or more of the preceding element.
# The matching should cover the entire input string (not partial).


# Example 1:

# Input: s = "aa", p = "a"
# Output: false
# Explanation: "a" does not match the entire string "aa".
# Example 2:

# Input: s = "aa", p = "a*"
# Output: true
# Explanation: '*' means zero or more of the preceding element, 'a'. Therefore, by repeating 'a' once, it becomes "aa".
# Example 3:

# Input: s = "ab", p = ".*"
# Output: true
# Explanation: ".*" means "zero or more (*) of any character (.)".
 

# Constraints:

# 1 <= s.length <= 20
# 1 <= p.length <= 20
# s contains only lowercase English letters.
# p contains only lowercase English letters, '.', and '*'.
# It is guaranteed for each appearance of the character '*', there will be a previous valid character to match.

# Sol_10}

class Solution:
    def isMatch(self, s: str, p: str) -> bool:
        memo = {}

        def dfs(i, j):
            if (i, j) in memo:
                return memo[(i, j)]

            if j >= len(p):
                return i >= len(s)

            match = i < len(s) and (s[i] == p[j] or p[j] == ".")
            
            if (j + 1) < len(p) and p[j + 1] == "*":
                memo[(i, j)] = dfs(i, j + 2) or (match and dfs(i + 1, j))
                return memo[(i, j)]
            if match:
                memo[(i, j)] = dfs(i + 1, j + 1)
                return memo[(i, j)]

            memo[(i, j)] = False
            return False

        return dfs(0, 0)