from src.utils.log import warn
from src.utils.utils import pretty_table_str


class Tuple(dict):
	def __init__(self, record, attrs=None, **kwargs):
		super(Tuple, self).__init__(**kwargs)

		if attrs is None:
			attrs = record._fields

		for attr in attrs:
			assert attr != "attrs", "Attribute name 'attrs' is not allowed"

			if isinstance(record, dict) and attr in record:
				self[attr] = record[attr]

			elif hasattr(record, attr):
				self[attr] = getattr(record, attr)

			else:
				warn("Skipping attribute '{}': not found in record {}".format(attr, record))
				raise Exception("Attribute '{}' not found in record {}".format(attr, record))


	def __getattr__(self, attr):
		return self[attr]


	def __setattr__(self, attr, value):
		self[attr] = value


	@property
	def attrs(self):
		return self.keys()


	def __str__(self):
		return pretty_table_str([self], header=self.attrs)


	def __iter__(self):
		return self.itervalues()



class Repr(Tuple):
	pass
