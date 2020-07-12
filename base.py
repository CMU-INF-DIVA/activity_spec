from enum import IntEnum

# Beginning part aligned with detectors.ObjectType
ProposalType = IntEnum('ProposalType', [
    'Vehicle', 'Person', 'Bike', 'PersonVehicle'], module=__name__)

ActivityType = IntEnum('ActivityType', [
    'person_abandons_package', 'person_closes_facility_door',
    'person_closes_trunk', 'person_closes_vehicle_door',
    'person_embraces_person', 'person_enters_scene_through_structure',
    'person_enters_vehicle', 'person_exits_scene_through_structure',
    'person_exits_vehicle', 'hand_interacts_with_person',
    'person_carries_heavy_object', 'person_interacts_with_laptop',
    'person_loads_vehicle', 'person_transfers_object',
    'person_opens_facility_door', 'person_opens_trunk',
    'person_opens_vehicle_door', 'person_talks_to_person',
    'person_picks_up_object', 'person_purchases', 'person_reads_document',
    'person_rides_bicycle', 'person_puts_down_object', 'person_sits_down',
    'person_stands_up', 'person_talks_on_phone', 'person_texts_on_phone',
    'person_steals_object', 'person_unloads_vehicle',
    'vehicle_drops_off_person', 'vehicle_picks_up_person', 'vehicle_reverses',
    'vehicle_starts', 'vehicle_stops', 'vehicle_turns_left',
    'vehicle_turns_right', 'vehicle_makes_u_turn'], module=__name__)
