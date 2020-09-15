from django.db import models
from typing import Iterable

class ListField(models.TextField):
    """
    A custom Django field to represent lists as comma separated strings
    """

    def __init__(self, *args, **kwargs):
        self.token = kwargs.pop('token', ',')
        super().__init__(*args, **kwargs)

    def deconstruct(self):
        name, path, args, kwargs = super().deconstruct()
        kwargs['token'] = self.token
        return name, path, args, kwargs

    def to_python(self, value):

        class SubList(list):
            def __init__(self, token, *args):
                self.token = token
                super().__init__(*args)

            def __str__(self):
                return self.token.join(self)

        if isinstance(value, list):
            return value
        if value is None:
            return SubList(self.token)
        return SubList(self.token, value.split(self.token))

    def from_db_value(self, value, expression, connection):
        return self.to_python(value)

    def get_prep_value(self, value):
        if not value:
            return
        assert(isinstance(value, Iterable))
        return self.token.join(value)

    def value_to_string(self, obj):
        value = self.value_from_object(obj)
        return self.get_prep_value(value)

PREPS = ["about", "above", "across", "after", "afterward", "against", "ago", "among", "apart", "around", "as", "at",
         "atop", "away", "back", "before", "behind", "below", "besides", "between", "beyond", "but", "by", "circa",
         "despite", "down", "downstairs", "during", "except", "for", "from", "home", "in", "indoors", "inside", "into",
         "like", "minus", "near", "nearby", "of", "off", "on", "onto", "out", "outside", "over", "past", "per", "plus",
         "round", "since", "than", "through", "throughout", "till", "to", "together", "toward", "under", "unlike",
         "until", "up", "upon", "via", "with", "within", "without"]

class Query(models.Model):
    # core data
    query_sentences = ListField(token='熕')
    preposition = models.CharField(choices=[(x, x) for x in PREPS], max_length=999)
    results = models.TextField()

    # metadata
    description = models.TextField()
    submitted_time = models.DateTimeField()

    # status tracking
    PENDING = 'PND'
    COMPLETE = 'CMP'
    STATUS_CHOICES = [(PENDING, 'Pending'), (COMPLETE, 'Complete')]
    status = models.CharField(max_length=3, choices=STATUS_CHOICES, default=PENDING)


def handle_query_result(task):
    query_id, result = task.result
    query = Query.objects.get(id=query_id)

    html = """
    <table class="table">
      <thead>
        <tr>
          <th scope="col">#</th>
          <th scope="col">Similarity</th>
          <th scope="col">Sentence</th>
        </tr>
      </thead>
      <tbody>
    """
    for i, (sim, sent) in enumerate(result):
        html += f'''    <tr>
      <th scope="row">{i+1}</th>
      <td>{sim}</td>
      <td>{sent}
    </tr>'''
    html += """  </tbody></table>"""
    query.status = Query.COMPLETE
    query.results = html
    query.save()


