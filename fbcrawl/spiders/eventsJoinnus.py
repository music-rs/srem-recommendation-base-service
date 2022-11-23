# -*- coding: utf-8 -*-
import scrapy
import logging

from scrapy.loader import ItemLoader
from scrapy.http import FormRequest
from scrapy.exceptions import CloseSpider
from scrapy.spiders import CSVFeedSpider
from fbcrawl.items import EventsGeneralItem, parse_date, parse_date2
from datetime import datetime
import json

class EventsJoinnusSpider(scrapy.Spider):
    name = 'eventsJoinnus'
    custom_settings = {
        'FEED_EXPORT_FIELDS': ['nombre','ubicacion','ubicacion_detalle','ubicacion_referencia','distrito','region', \
                               'fecha_inicio','fecha_inicio_alt','fecha_fin','descripcion','precio','organizador','asistiran','me_interesa','veces_compartido','imagen','enlace_evento'],
        'DUPEFILTER_CLASS' : 'scrapy.dupefilters.BaseDupeFilter',
        'CONCURRENT_REQUESTS' : 1
    }
    allowed_domains = ['joinnus.com']
    quotes_base_url = 'https://www.joinnus.com/categoryc/concerts/PE/%s'
    start_urls = [quotes_base_url % 1]
    download_delay = 2
    #start_urls = ['https://www.joinnus.com/categoryc/concerts/PE']

    def parse(self, response):

        #data = json.loads(response.body)

        #//div[contains(@class, "row")] -> los eventos de musica proximos
        EVENTOS_XPATH='//div[contains(@class, "activity")]'
        #Recorremos cada uno de los eventos de la Pagina de JOINNUS
        for event in response.xpath(EVENTOS_XPATH):

            url = event.xpath('.//div/div[1]/a/@href').extract_first()
            print (url)
            yield response.follow(url, callback=self.parse_event)

        next_page = response.xpath('//a[contains(@class, "more-link")]/@href').get()
        if next_page is not None:
            yield response.follow(next_page, self.parse)


    def parse_event(self, response):
        EVENT_NAME='//h1/strong/text()'
        EVENT_WHERE='//div[contains(@id, "ubicacion")]/p[1]/text()'
        EVENT_REGION='//div[contains(@id, "ubicacion")]/h1/text()'
        EVENT_UBICACION_REFERENCIA='//div[contains(@id, "ubicacion")]/p[2]/text()'
        EVENT_ORGANIZADOR='//div[contains(@id, "autor")]/div/span/a[1]/text()'
        EVENT_FECHA='//div[contains(@id, "formulario")]/form/div/div[1]/strong/text()'
        EVENT_HORA='//div[contains(@id, "formulario")]/form/div/div/div/b/text()'
        EVENT_TIPO_PRECIO='//td[contains(@class, "paymentcell")]/text()'
        EVENT_PRECIO='//span[contains(@class, "amount")]/text()'
        EVENT_DESCRIPTION='//p[contains(@style, "text-align")][1]/text()[1]'
        EVENT_IMAGE='//img[contains(@id, "fotodeevento")]/@src'

        #Obtener nombre del evento
        self.nombre = response.xpath(EVENT_NAME).extract_first()

        #Obtener ubicación del evento
        self.ubicacion = response.xpath(EVENT_WHERE).extract_first()
        self.region = response.xpath(EVENT_REGION).extract_first()
        self.ubicacion_referencia = response.xpath(EVENT_UBICACION_REFERENCIA).extract_first()

        #Obtener el organizador del evento
        self.organizador = response.xpath(EVENT_ORGANIZADOR).extract_first()

        #Obtener fecha del evento
        date = response.xpath(EVENT_FECHA).extract_first()
        hora = response.xpath(EVENT_HORA).extract_first()

        dateArray = date.split('al')
        dateAño = date.split(',')[0]

        self.fecha_inicio = "{} - {}, {}".format(dateArray[0].strip() , hora.strip(),dateAño.strip())  #quitar espacios al final

        if len(dateArray) == 2:
            self.fecha_fin = "{}, {}".format(dateArray[1].strip(), dateAño.strip())
            #self.fecha_inicio = dateArray[0].strip()
            #self.fecha_fin = dateArray[1].strip()
        else:
            self.fecha_fin = "{}, {}".format(dateArray[0].strip(), dateAño.strip())
            #self.fecha_inicio = date
            #self.fecha_fin = date

        #Obtener precios
        self.precio = response.xpath(EVENT_PRECIO).getall()
        tipoPrecio = response.xpath(EVENT_TIPO_PRECIO).getall()

        print('***********************')
        print(self.precio)
        print(tipoPrecio)

        #Obtener la descripcion del evento
        self.descripcion = response.xpath(EVENT_DESCRIPTION).extract_first()

        #Obtener la imagen del evento
        self.imagen = response.xpath(EVENT_IMAGE).extract_first()
        
        yield EventsGeneralItem(
            nombre=self.nombre,
            ubicacion=self.ubicacion,
            ubicacion_detalle=self.ubicacion,
            ubicacion_referencia=self.ubicacion_referencia,
            distrito = '',
            region = self.region,
            fecha_inicio=self.fecha_inicio,
            fecha_inicio_alt='',
            fecha_fin=self.fecha_fin,
            descripcion=self.descripcion,
            precio = self.precio,
            organizador = self.organizador,
            asistiran = '',
            me_interesa = '',
            veces_compartido = '',
            imagen=self.imagen,
            enlace_evento=response.url
        )