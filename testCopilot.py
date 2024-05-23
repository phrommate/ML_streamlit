# calculate days between two dates
def daysBetweenDates(year1, month1, day1, year2, month2, day2):
    # calculate days in year
    def daysInYear(year):
        if year % 4 != 0:
            return 365
        elif year % 100 != 0:
            return 366
        elif year % 400 != 0:
            return 365
        else:
            return 366

    # calculate days in month
    def daysInMonth(year, month):
        if month == 1 or month == 3 or month == 5 or month == 7 or month == 8 or month == 10 or month == 12:
            return 31
        elif month == 4 or month == 6 or month == 9 or month == 11:
            return 30
        elif daysInYear(year) == 366:
            return 29
        else:
            return 28

    # calculate days in year1
    days = 0
    for y in range(year1, year2):
        days += daysInYear(y)

    # calculate days in month1
    for m in range(1, month1):
        days += daysInMonth(year1, m)

    # calculate days in day1
    days += day1

    # calculate days in year2
    for m in range(1, month2):
        days += daysInMonth(year2, m)

    # calculate days in month2
    days += day2

    return days

# Call function
print(daysBetweenDates(2024, 1, 1, 2024, 2, 28)) # 58
print(daysBetweenDates(2024, 1, 1, 2024, 1, 30)) # 60

