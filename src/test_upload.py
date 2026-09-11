import requests

url = "http://localhost:8000/listings/upload"

files = {
    "image": open("data/listings/seller_b/stolen_from_a_0.jpg", "rb")
}
data = {
    "seller_id": "seller_c",
    "listing_title": "Brand New Shimano Shifter",
    "listing_description": "brand new, never installed, sealed in box",
    "listing_category": "Bike Parts",
}

response = requests.post(url, files=files, data=data)
print(response.status_code)
print(response.json())