import scrapy

from scrapy.loader import ItemLoader
from scrapy.exceptions import CloseSpider
from fbcrawl.spiders.fbcrawl import FacebookSpider
from fbcrawl.items import EventsGeneralItem, parse_date, parse_date2

from datetime import datetime

class EventsGeneralFBSpider(FacebookSpider):
    """
    Parse FB events, given a page (needs credentials)
    """
    name = "eventsGeneralFB"
    custom_settings = {
        'FEED_EXPORT_FIELDS': ['nombre','ubicacion','ubicacion_detalle','ubicacion_referencia','distrito','region', \
                               'fecha_inicio','fecha_inicio_alt','fecha_fin','descripcion','precio','organizador','asistiran','me_interesa','veces_compartido','imagen','enlace_evento'],
        'DUPEFILTER_CLASS' : 'scrapy.dupefilters.BaseDupeFilter',
        'CONCURRENT_REQUESTS' : 1
    }

    def __init__(self, *args, **kwargs):
        self.page = kwargs['page']
        self.pageOriginal = kwargs['page']
        super().__init__(*args,**kwargs)

#//div[contains(@class, "bx")]

    def parse_page(self, response):
        TABLE_XPATH='//html/body/div/div/div[2]/div/table/tbody/tr/td/div[1]/table/tbody/tr/td'
        for event in response.xpath(TABLE_XPATH):
            url = event.xpath('.//a/@href').extract_first()
            yield response.follow(url, callback=self.parse_events)

    def parse_events(self, response):
        TABLE_XPATH='//html/body/div/div/div[2]/div/table/tbody/tr/td/div[2]/div/div'
        for event in response.xpath(TABLE_XPATH):
            url = event.xpath('.//a/@href').extract_first()
            yield response.follow(url, callback=self.parse_event)



    def parse_event(self, response):

        EVENT_NAME='/html/body/div/div/div[2]/div/table/tbody/tr/td/div[2]/div[2]/div[1]/h3/text()'
        EVENT_NAME2='/html/body/div/div/div[2]/div/table/tbody/tr/td/div[2]/div[1]/div[1]/h3/text()'
        EVENT_WHERE='/html/body/div/div/div[2]/div/table/tbody/tr/td/div[3]/div/div[2]/table/tbody/tr/td[2]/dt/div/text()'
        EVENT_LOCATION='/html/body/div/div/div[2]/div/table/tbody/tr/td/div[3]/div/div[2]/table/tbody/tr/td[2]/dd/div/text()'
        DATE='/html/body/div/div/div[2]/div/table/tbody/tr/td/div[3]/div/div[1]/table/tbody/tr/td[2]/dt/div/text()'
        EVENT_DESCRIPTION='/html/body/div/div/div[2]/div/table/tbody/tr/td/table/tbody/tr/td/div[2]/div[2]/div[2]/div[2]/text()'
        EVENT_COVER='/html/body/div/div/div[2]/div/table/tbody/tr/td/div[2]/div[1]/a/img/@src'
        EVENT_ORGANIZADOR = '//a[contains(@class, "_4e81")]/text()' 
        EVENT_ORGANIZADOR2 = '//a[contains(@class, "by")]/text()'
        EVENT_ASISTIRAN = ' /html/body/div/div/div[2]/div/table/tbody/tr/td/table/tbody/tr/td/div[1]/div[2]/div[1]/div/div[1]/div[2]/a/text()'
        EVENT_ASISTIRAN2 = '/html/body/div/div/div[2]/div/table/tbody/tr/td/table/tbody/tr/td/div[2]/div[2]/div[1]/div/div[1]/div[2]/a/text()'
        EVENT_ME_INTERESA = '/html/body/div/div/div[2]/div/table/tbody/tr/td/table/tbody/tr/td/div[1]/div[2]/div[1]/div/div[2]/div[2]/a/text()'
        EVENT_ME_INTERESA2 = '/html/body/div/div/div[2]/div/table/tbody/tr/td/table/tbody/tr/td/div[2]/div[2]/div[1]/div/div[2]/div[2]/a/text()'
        EVENT_VECES_COMPARTIDO = '/html/body/div/div/div[2]/div/table/tbody/tr/td/table/tbody/tr/td/div[1]/div[2]/div[1]/div/div[3]/div[2]/div/text()'
        EVENT_VECES_COMPARTIDO2 = '/html/body/div/div/div[2]/div/table/tbody/tr/td/table/tbody/tr/td/div[2]/div[2]/div[1]/div/div[3]/div[2]/div/text()'
        EVENT_ASISTIRAN_ME_INTERESA_VECES_COMPARTIDO = '/html/body/div/div/div[2]/div/table/tbody/tr/td/table/tbody/tr/td/div[1]/div[2]/div[1]/div[1]/a/div/text()'
        EVENT_ASISTIRAN_ME_INTERESA_VECES_COMPARTIDO2 = '/html/body/div/div/div[2]/div/table/tbody/tr/td/table/tbody/tr/td/div[2]/div[2]/div[1]/div[1]/a/div/text()'

        date = response.xpath(DATE).extract_first()

        name1 = response.xpath(EVENT_NAME).extract_first()
        name2 = response.xpath(EVENT_NAME2).extract_first()
        self.name = ""

        if name1 and len(name1) > 0:
            self.name = name1
        elif name2 and len(name2) > 0:
            self.name = name2

        self.logger.info('Parsing event %s' % self.name)

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
        
        #OBTENER EL ORGANIZADOR DEL EVENTO
        organizador1 = response.xpath(EVENT_ORGANIZADOR).extract_first()
        organizador2 = response.xpath(EVENT_ORGANIZADOR2).extract_first()

        self.organizador = ""

        if organizador1 and len(organizador1) > 0:
            self.organizador = organizador1
        elif organizador2 and len(organizador2) > 0:
            self.organizador = organizador2

        #Obtener el interes de las personas del evento
        asistiran1 = response.xpath(EVENT_ASISTIRAN).extract_first()
        asistiran2 = response.xpath(EVENT_ASISTIRAN2).extract_first()
        me_interesa1= response.xpath(EVENT_ME_INTERESA).extract_first()
        me_interesa2 = response.xpath(EVENT_ME_INTERESA2).extract_first()
        veces_compartido1 = response.xpath(EVENT_VECES_COMPARTIDO).extract_first()
        veces_compartido2 = response.xpath(EVENT_VECES_COMPARTIDO2).extract_first()
        intereses1 = response.xpath(EVENT_ASISTIRAN_ME_INTERESA_VECES_COMPARTIDO).extract_first()
        intereses2 = response.xpath(EVENT_ASISTIRAN_ME_INTERESA_VECES_COMPARTIDO2).extract_first()

        self.asistiran = ""
        self.me_interesa = ""
        self.veces_compartido = ""

        print (asistiran1,' ******* ', asistiran2)

        if asistiran1 and len(asistiran1) > 0:
            self.asistiran = asistiran1
        elif asistiran2 and len(asistiran2) > 0:
            self.asistiran = asistiran2

        if me_interesa1 and len(me_interesa1) > 0:
            self.me_interesa = me_interesa1
        elif me_interesa2 and len(me_interesa2) > 0:
            self.me_interesa = me_interesa2

        if veces_compartido1 and len(veces_compartido1) > 0:
            self.veces_compartido = veces_compartido1
        elif veces_compartido2 and len(veces_compartido2) > 0:
            self.veces_compartido = veces_compartido2

        #33 asistirán · 208 interesados · 9 veces compartido    237 asistirán · 1.8 mil interesados · 77 veces compartido
        if self.asistiran == "" and self.me_interesa == "" and self.veces_compartido == "":
            if intereses1 and len(intereses1) > 0:
                self.asistiran = intereses1.split('·')[0].replace('asistirán', '').replace('&nbsp;', '').replace(' ', '').strip()
                self.me_interesa = intereses1.split('·')[1].replace('interesados', '').replace('&nbsp;', '').replace(' ', '').strip()
                self.veces_compartido = intereses1.split('·')[2].replace('veces compartido', '').replace('&nbsp;', '').replace(' ', '').strip()
            elif intereses2 and len(intereses2) > 0:
                self.asistiran = intereses2.split('·')[0].replace('asistirán', '').replace('&nbsp;', '').replace(' ', '').strip()
                self.me_interesa = intereses2.split('·')[1].replace('interesados', '').replace('&nbsp;', '').replace(' ', '').strip()
                self.veces_compartido = intereses2.split('·')[2].replace('veces compartido', '').replace('&nbsp;', '').replace(' ', '').strip()

        yield EventsGeneralItem(
            nombre=self.name,
            ubicacion=response.xpath(EVENT_WHERE).extract_first(),
            ubicacion_detalle=response.xpath(EVENT_LOCATION).extract_first(),
            ubicacion_referencia='',
            distrito = '',
            region = '',
            fecha_inicio=start_date,
            fecha_inicio_alt='',
            fecha_fin=end_date,
            descripcion=response.xpath(EVENT_DESCRIPTION).getall(),
            precio = '',
            organizador = self.organizador,
            asistiran = self.asistiran,
            me_interesa = self.me_interesa,
            veces_compartido = self.veces_compartido,
            imagen=response.xpath(EVENT_COVER).extract_first(),
            enlace_evento=response.url
        )
