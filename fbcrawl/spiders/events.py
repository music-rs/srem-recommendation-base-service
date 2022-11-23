import scrapy

from scrapy.loader import ItemLoader
from scrapy.exceptions import CloseSpider
from fbcrawl.spiders.fbcrawl import FacebookSpider
from fbcrawl.items import EventsItem, parse_date, parse_date2

from datetime import datetime

class EventsSpider(FacebookSpider):
    """
    Parse FB events, given a page (needs credentials)
    """
    name = "events"
    custom_settings = {
        'FEED_EXPORT_FIELDS': ['name','where','location','photo','start_date', \
                               'end_date','description','enlace'],
        'DUPEFILTER_CLASS' : 'scrapy.dupefilters.BaseDupeFilter',
        'CONCURRENT_REQUESTS' : 1
    }

    def __init__(self, *args, **kwargs):
        self.page = kwargs['page']
        self.pageOriginal = kwargs['page']
        super().__init__(*args,**kwargs)

    def parse_page(self, response):
        yield scrapy.Request(url=response.urljoin('%sevents' % self.pageOriginal),
                             callback=self.parse_events,
                             priority=10,
                             meta={'index':1})

    def parse_events(self, response):
        TABLE_XPATH='/html/body/div/div/div[2]/div/table/tbody/tr/td/div[2]/div/div/div[2]/div/table/tbody/tr'
        for event in response.xpath(TABLE_XPATH):
            url = event.xpath('.//td/div/div/span[3]/div/a[1]/@href').extract_first()
            yield response.follow(url, callback=self.parse_event)

    def parse_event(self, response):
        EVENT_NAME='/html/body/div/div/div[2]/div/table/tbody/tr/td/div[2]/div[2]/div[1]/h3/text()'
        EVENT_WHERE='/html/body/div/div/div[2]/div/table/tbody/tr/td/div[3]/div/div[2]/table/tbody/tr/td[2]/dt/div/text()'
        EVENT_LOCATION='/html/body/div/div/div[2]/div/table/tbody/tr/td/div[3]/div/div[2]/table/tbody/tr/td[2]/dd/div/text()'
        DATE='/html/body/div/div/div[2]/div/table/tbody/tr/td/div[3]/div/div[1]/table/tbody/tr/td[2]/dt/div/text()'
        EVENT_DESCRIPTION='/html/body/div/div/div[2]/div/table/tbody/tr/td/table/tbody/tr/td/div[2]/div[2]/div[2]/div[2]/text()'
        EVENT_COVER='/html/body/div/div/div[2]/div/table/tbody/tr/td/div[2]/div[1]/a/img/@src'
        date = response.xpath(DATE).extract_first()

        if self.lang == 'es':
            #print('############################')
            #print(date,len (date.split('–')),len (date.split('de')))

            #date = Viernes, 23 de agosto de 2019 de 22:00 a 4:00
            if (len (date.split('de')) == 4):
                event_date = "{} de {} de {}".format(date.split('de')[0],date.split('de')[1],date.split('de')[2])    #Viernes, 23 de agosto de 2019 
                hours_date = date.split('de')[3]    #22:00 a 4:00

                start_hour = hours_date.split('a')[0].replace(' ', '') or None  #22:00
                end_hour = hours_date.split('a')[1].replace(' ', '') or None    #4:00

                start_date = "{} - {} ".format(event_date, start_hour)
                end_date = "{} - {} ".format(event_date, end_hour)

            #date = Sábado, 23 de noviembre de 2019 a las 21:00
            elif (len (date.split('de')) == 3 and len (date.split('–')) == 1):
                event_date = date.split('a las')[0]                             #Sábado, 23 de noviembre de 2019 
                start_hour = date.split('a las')[1].replace(' ', '') or None    #21:00

                start_date = "{} - {} ".format(event_date, start_hour)
                end_date = ""

            #date = jueves de 21:00 a 5:00
            elif (len (date.split('de')) == 2):     
                event_date = datetime.now()         #Fecha de hoy 
                hours_date = date.split('de')[1]    #21:00 a 5:00

                start_hour = hours_date.split('a')[0].replace(' ', '') or None  #22:00
                end_hour = hours_date.split('a')[1].replace(' ', '') or None    #4:00

                start_date = "{} - {} ".format(event_date, start_hour)
                end_date = "{} - {} ".format(event_date, end_hour)

            #Hoy a las 22:00
            elif (len (date.split('de')) == 1):
                event_date = datetime.now()                                 #Fecha de hoy 
                start_hour = date.split('las')[1].replace(' ', '') or None    #22:00

                start_date = "{} - {} ".format(event_date, start_hour)
                end_date = ""

            #date = 12 de oct., 17:00 – 13 de oct., 3:00
            elif (len (date.split('–')) == 2 ):  
                start_date = date.split('–')[0]
                end_date = date.split('–')[1]

        else: 
            start_date = date
            end_date = date
        
        name = response.xpath(EVENT_NAME).extract_first()
        self.logger.info('Parsing event %s' % name)
        yield EventsItem(
            name=name,
            where=response.xpath(EVENT_WHERE).extract_first(),
            location=response.xpath(EVENT_LOCATION).extract_first(),
            photo=response.xpath(EVENT_COVER).extract_first(),
            start_date=start_date,
            end_date=end_date,
            description=response.xpath(EVENT_DESCRIPTION).getall(),
            enlace=response.url
        )
