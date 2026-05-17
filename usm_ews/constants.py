LABEL_DECODER = {0: "Dropout", 1: "Enrolled", 2: "Graduate"}

MARITAL_STATUS = {
    "Single": 1,
    "Married": 2,
    "Widower": 3,
    "Divorced": 4,
    "Facto Union / Live-in": 5,
    "Legally Separated": 6,
}

APPLICATION_MODE = {
    "Regular Admission (Freshman)": 1,
    "Transferee": 42,
    "Cross-Enrollee": 43,
    "Change of Course": 51,
    "Irregular Student": 17,
}

COURSE = {
    "BS Agriculture": 9003,
    "BS Agronomy": 9070,
    "BS Horticulture": 9085,
    "BS Agricultural and Biosystems Engineering": 9119,
    "BS Civil Engineering": 9130,
    "BS Computer Engineering": 9147,
    "BS Electronics Engineering": 9238,
    "BS Computer Science": 9254,
    "BS Information Systems": 9500,
    "BS Nursing": 9556,
    "BS Midwifery": 9670,
    "BS Pharmacy": 9773,
    "BS Nutrition and Dietetics": 9853,
    "Bachelor of Elementary Education (BEEd)": 8014,
    "Bachelor of Early Childhood Education": 171,
    "Bachelor of Secondary Education (BSEd)": 33,
    "BS Forestry": 9991,
    "BS Environmental Science": 9238,
    "Bachelor of Library and Information Science": 9119,
    "Doctor of Veterinary Medicine (DVM)": 9130,
}

PREV_QUALIFICATION = {
    "Senior High School Graduate (Academic Track)": 1,
    "Senior High School Graduate (TVL Track)": 9,
    "Senior High School Graduate (Arts & Design / Sports)": 12,
    "ALS Passer": 10,
    "TESDA / TVET Certificate": 39,
    "College Undergraduate (did not finish)": 6,
    "Bachelor's Degree Graduate": 2,
    "Bachelor's Degree (Different Field)": 3,
    "Master's Degree": 4,
    "Doctorate": 5,
    "Elementary Graduate (old curriculum)": 11,
    "High School Graduate (old curriculum)": 14,
}

QUALIFICATION_LEVEL = {
    "Elementary Graduate": 35,
    "High School Graduate (old curriculum)": 14,
    "Senior High School Graduate": 1,
    "ALS Passer": 10,
    "TESDA / TVET Certificate (NC I-III)": 39,
    "College Undergraduate (did not finish)": 6,
    "Bachelor's Degree": 2,
    "Bachelor's Degree (different field)": 3,
    "Post-Baccalaureate / Professional Course": 41,
    "Master's Degree": 4,
    "Doctorate": 5,
    "Can't Read or Write": 35,
    "Unknown / Not Stated": 34,
}

OCCUPATION = {
    "Student / Not Working": 0,
    "Farmer / Fisherfolk": 6,
    "Skilled Agricultural / Forestry Worker": 163,
    "Government Employee (Rank & File)": 4,
    "Government Employee (Supervisor/Manager)": 1,
    "Private Employee (Rank & File)": 8,
    "Private Employee (Supervisor/Manager)": 112,
    "Teacher / Education Professional": 123,
    "Health Professional (Doctor/Nurse/Midwife)": 122,
    "Engineer / Technician": 121,
    "OFW / Overseas Worker": 183,
    "Self-Employed / Small Business Owner": 152,
    "Driver / Transport Worker": 182,
    "Construction Worker / Laborer": 171,
    "Factory / Assembly Worker": 181,
    "Domestic Helper / Household Service Worker": 151,
    "Market Vendor / Street Seller": 195,
    "Armed Forces / PNP": 101,
    "Retired / Not Working": 90,
    "Deceased / Unknown": 99,
}

BINARY = {"No": 0, "Yes": 1}
GENDER = {"Female": 0, "Male": 1}
APPLICATION_ORDER = {"1st Choice Program": 1, "2nd Choice Program": 2}

HOUSEHOLD_INCOME = {
    "Class E - Below P10,000 / month": 0.5,
    "Class D - P10,000 to P20,000 / month": 1.2,
    "Class C - P20,000 to P50,000 / month": 2.5,
    "Class B - P50,000 to P100,000 / month": 4.0,
    "Class A - Above P100,000 / month": 6.0,
}

COLOR_MAP = {
    "Dropout": "#ef5350",
    "Enrolled": "#ff9800",
    "Graduate": "#66bb6a",
    "Unknown": "#999999",
}

CLASS_MAP = {
    "Dropout": "dropout",
    "Enrolled": "enrolled",
    "Graduate": "graduate",
    "Unknown": "unknown",
}
