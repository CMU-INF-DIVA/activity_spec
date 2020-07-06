from typing import List

import pandas as pd

ProposalRegistry = {}


class Proposal(object):

    RECORD = ['type']

    def to_record(self):
        return {'type': 'Proposal'}

    @classmethod
    def parse_record(cls, record):
        assert record['type'] == cls.__name__
        values = [record[k] for k in cls.RECORD[1:]]
        return values

    def __repr__(self):
        record_str = ', '.join([
            '%s=%s' % (k, v) for k, v in self.to_record().items()
            if k != 'type'])
        return '%s(%s)' % (self.__class__.__name__, record_str)


class Collection(object):

    def __init__(self, proposals: List[Proposal]):
        self.proposals = proposals

    def __len__(self):
        return len(self.proposals)

    def to_record(self):
        record = [p.to_record() for p in self.proposals]
        record = pd.DataFrame(record)
        return record

    @classmethod
    def parse_record(cls, record):
        record = record.to_dict('records')
        proposals = [ProposalRegistry[r['type']].from_record(r)
                     for r in record]
        return proposals

    def save_record(self, filename):
        record = self.to_record()
        record.to_csv(filename)

    @classmethod
    def load_record(cls, filename):
        record = pd.read_csv(filename)
        return record

    def __repr__(self):
        return '%s(size=%d)' % (self.__class__.__name__, len(self))
