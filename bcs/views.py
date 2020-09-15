import datetime

from django.core.exceptions import ValidationError
from django.shortcuts import render, redirect
from django import forms
from crispy_forms.helper import FormHelper
from crispy_forms.layout import Layout, Div, Submit, HTML, Button, Row, Field
from crispy_forms.bootstrap import AppendedText, PrependedText, FormActions
from nltk.tokenize import word_tokenize
import bcs.models as mdl
from django_q.tasks import async_task, result


class QueryForm(forms.Form):
    def clean(self):
        cleaned_data = super().clean()
        prep = self.data['preposition']
        sents = [s for s in cleaned_data['query_sentences'].replace('\r', '').split('\n') if len(s.strip()) > 0]

        if any(len(s) < 8 for s in sents):
            self.add_error('query_sentences', 'Sentences must be at least 8 chars long')
        if any(len(s) > 120 for s in sents):
            self.add_error('query_sentences', 'Sentences must be less than 120 chars long')
        for i, s in enumerate(sents):
            tokens = word_tokenize(s)
            print(tokens)
            if len([t for t in tokens if t.lower() == prep.lower()]) != 1:
                self.add_error('query_sentences', f'Sentence {i+1}: must contain exactly one occurrence of the preposition')

        if len(self.data['description']) <= 10:
            self.add_error('description', 'Please write an informative description, >10 chars in length')

        return {"preposition": prep,
                "query_sentences": "\n".join(s.strip() for s in sents),
                "description": self.data['description']}

    preposition = forms.ChoiceField(
        choices=(
            (x, x) for x in mdl.PREPS
        ),
        widget=forms.Select,
        #help_text = "<strong>Note:</strong> Labels surround all the options for much larger click areas and a more usable form.",
    )

    description = forms.CharField()

    query_sentences = forms.CharField(
        widget=forms.Textarea,
        help_text="<strong><em>Note:</em></strong> each sentence must be newline-separated, and must contain the target preposition exactly once"
    )

    helper = FormHelper()
    helper.form_class = 'form-horizontal'
    helper.layout = Layout(
        Field('preposition'),
        Field('description'),
        Field('query_sentences', rows="3"),
    )


def index(request):
    return redirect('results')


def query(request):
    if request.method == 'POST':
        form = QueryForm(request.POST)
        print('valid?', form.is_valid())
        if form.is_valid():
            data = form.clean()
            prep = data['preposition']
            sents = data['query_sentences'].split('\n')
            query = mdl.Query.objects.create(
                preposition=prep,
                query_sentences=sents,
                description=data['description'],
                status=mdl.Query.PENDING,
                submitted_time=datetime.datetime.now()
            )
            async_task('bcs.encow.query_encow', query.id, prep, sents, hook='bcs.models.handle_query_result')
            return redirect('results')
    else:
        form = QueryForm()

    return render(request, 'search.html', {'form': form})


def results(request):
    context = {}
    context['queries'] = mdl.Query.objects.order_by('-id')
    for q in context['queries']:
        q.status = q.get_status_display()
    return render(request, 'results.html', context)


def results_detail(request, id):
    query = mdl.Query.objects.get(id=id)
    return render(request, 'results_detail.html', {'id': id, 'query': query})
