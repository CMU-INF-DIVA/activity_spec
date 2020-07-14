from enum import IntEnum, auto


class ProposalType(IntEnum):

    Unknown = -1
    
    # Aligned with detectors.ObjectType
    Vehicle = 1
    Person = auto()
    Bike = auto()

    VehiclePerson = 11
    PersonPerson = auto()


class ActivityTypeSDL(IntEnum):

    Negative = -1
    Ignore = 0
    person_abandons_package = auto()
    person_closes_facility_door = auto()
    person_closes_trunk = auto()
    person_closes_vehicle_door = auto()
    person_embraces_person = auto()
    person_enters_scene_through_structure = auto()
    person_enters_vehicle = auto()
    person_exits_scene_through_structure = auto()
    person_exits_vehicle = auto()
    hand_interacts_with_person = auto()
    person_carries_heavy_object = auto()
    person_interacts_with_laptop = auto()
    person_loads_vehicle = auto()
    person_transfers_object = auto()
    person_opens_facility_door = auto()
    person_opens_trunk = auto()
    person_opens_vehicle_door = auto()
    person_talks_to_person = auto()
    person_picks_up_object = auto()
    person_purchases = auto()
    person_reads_document = auto()
    person_rides_bicycle = auto()
    person_puts_down_object = auto()
    person_sits_down = auto()
    person_stands_up = auto()
    person_talks_on_phone = auto()
    person_texts_on_phone = auto()
    person_steals_object = auto()
    person_unloads_vehicle = auto()
    vehicle_drops_off_person = auto()
    vehicle_picks_up_person = auto()
    vehicle_reverses = auto()
    vehicle_starts = auto()
    vehicle_stops = auto()
    vehicle_turns_left = auto()
    vehicle_turns_right = auto()
    vehicle_makes_u_turn = auto()
