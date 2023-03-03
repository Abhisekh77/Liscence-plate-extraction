get(id):
returns:

{
"username": "username" <string>,
"license": url <IMG_URL>,
"registered": True <BOOL>
}

# Verify fingerprint : after license plate recognition

finger.verify():
return {
id: fingerprint.id <integer>,
valid: True <Boolean>
}

# Pseudo Code

record_number_plate()
input_no_of_passengers
finger_response = verify_fingerprint() # todo
fingerprint_id = finger_response.id if finger_response.valid else -1
if fingerprint_id == -1:
print("\n\n ------------------- driver not registered ------------------- \n\n")

# Validate data

user_data = requests.get("http://localhost:8000/api/v1/driver/11").json()
date = tuple([int(a) for a in user_data["expiry_date"].split('-')])
datetime.datetime(date[0], date[1], date[2])
if user_data.license_expiry < current_data:
print("\n\n ------------------- License Expired ------------------- \n\n")

# Todo:

- register user UI
- Verify fingerprint after read_number_plate
- verify_fingerprint backend
- Register fingerprint backend, UI

  username=request.username,
  license_img=request.licenseImg, expiry_date=request.expiryDate
  username anon
  license_img http://web.mit.edu/2.744/studentSubmissions/humanUseAnalysis/keval/vm.jpg
  expiry_date 11
