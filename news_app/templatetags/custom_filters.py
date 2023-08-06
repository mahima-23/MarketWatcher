from django import template

register = template.Library()

@register.filter
def filter_attribute(queryset, attr_string):
    attributes = attr_string.split(',')
    for attr in attributes:
        queryset = [obj for obj in queryset if getattr(obj, attr.strip(), '')]
    return queryset
