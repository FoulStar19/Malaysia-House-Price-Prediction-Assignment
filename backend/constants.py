"""Shared option lists and lookups used by the API.

Kept separate from main.py so both model_utils.py and main.py can import
without circular imports.
"""

FACILITY_OPTIONS = [
    "Barbeque area", "Club house", "Gymnasium", "Jogging Track", "Lift",
    "Minimart", "Multipurpose hall", "Parking", "Playground", "Sauna",
    "Security", "Squash Court", "Swimming Pool", "Tennis Court",
]

NEARBY_OPTIONS = [
    "Mall", "Park", "School", "Hospital", "Bus_Stop", "Highway",
    "Railway_Station", "Nearby_School", "Nearby_Mall", "Nearby_Railway_Station",
]

PROPERTY_TYPES = [
    "Apartment", "Condominium", "Service Residence", "Studio",
    "Duplex", "Flat", "Townhouse Condo", "Others",
]

TENURE_OPTIONS = ["Freehold", "Leasehold"]
LAND_OPTIONS = ["Non Bumi Lot", "Bumi Lot", "Malay Reserved"]
FLOOR_RANGE_OPTIONS = ["Low", "Medium", "High", "-"]

# The 10 states the model actually has one-hot columns for (State_<name>),
# each with an approximate centroid so the frontend can plot them on a map.
# "Other" has no sensible single point, so it's placed near the middle of
# the peninsula and excluded from the map layer by the frontend.
STATE_OPTIONS = [
    "Selangor", "Kuala Lumpur", "Johor", "Penang", "Melaka",
    "Negeri Sembilan", "Sabah", "Sarawak", "Putrajaya", "Other",
]

STATE_COORDS = {
    "Selangor": (3.0738, 101.5183),
    "Kuala Lumpur": (3.1390, 101.6869),
    "Johor": (1.4854, 103.7618),
    "Penang": (5.4141, 100.3288),
    "Melaka": (2.1896, 102.2501),
    "Negeri Sembilan": (2.7258, 101.9424),
    "Sabah": (5.9788, 116.0753),
    "Sarawak": (1.5533, 110.3592),
    "Putrajaya": (2.9264, 101.6964),
}
