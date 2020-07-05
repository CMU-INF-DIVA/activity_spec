class ActivityProposal(object):

    def __init__(self, start_frame_id, length, image_box):
        self.start_frame_id = start_frame_id
        self.length = length
        self.image_box = image_box
