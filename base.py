import os
import os.path as osp
from enum import IntEnum, auto


class ProposalType(IntEnum):

    Unknown = -1

    # Aligned with detectors.ObjectType
    Vehicle = 1
    Person = auto()
    Bike = auto()

    VehiclePerson = 11
    PersonPerson = auto()


class ActivityTypeMEVA(IntEnum):

    Negative = 0
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


class ActivityTypeVIRAT(IntEnum):

    Negative = 0
    person_closes_facility_or_vehicle_door = auto()
    person_closes_trunk = auto()
    vehicle_drops_off_person = auto()
    person_enters_facility_or_vehicle = auto()
    person_exits_facility_or_vehicle = auto()
    person_interacts_object = auto()
    person_loads_vehicle = auto()
    person_opens_trunk = auto()
    person_opens_facility_or_vehicle_door = auto()
    person_person_interaction = auto()
    person_pickups_object = auto()
    vehicle_picks_up_person = auto()
    person_pulls_object = auto()
    person_pushs_object = auto()
    person_rides_bicycle = auto()
    person_sets_down_object = auto()
    person_talks_to_person = auto()
    person_carries_heavy_object = auto()
    person_unloads_vehicle = auto()
    person_carries_object = auto()
    person_crouches = auto()
    person_gestures = auto()
    person_runs = auto()
    person_sits = auto()
    person_stands = auto()
    person_walks = auto()
    person_talks_on_phone = auto()
    person_texts_on_phone = auto()
    person_uses_tool = auto()
    vehicle_moves = auto()
    vehicle_starts = auto()
    vehicle_stops = auto()
    vehicle_turns_left = auto()
    vehicle_turns_right = auto()
    vehicle_makes_u_turn = auto()


ActivityTypes = {'MEVA': ActivityTypeMEVA, 'VIRAT': ActivityTypeVIRAT}


def load_activity_index(dataset):
    activity_index_dir = osp.join(
        os.environ['datasets_dir'], dataset, 'meta/activity-index/single')
    name = 'ActivityType' + dataset
    types = ['Negative'] + os.listdir(activity_index_dir)
    enum = IntEnum(name, types, start=0)
    globals()[name] = enum
    return enum


dataset = os.environ.get('dataset', None)
if dataset is not None and dataset not in ActivityTypes:
    ActivityTypes[dataset] = load_activity_index(dataset)
