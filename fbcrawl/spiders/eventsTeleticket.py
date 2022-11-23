# -*- coding: utf-8 -*-
import scrapy
import logging

from scrapy.loader import ItemLoader
from scrapy.http import FormRequest
from scrapy.exceptions import CloseSpider
from fbcrawl.items import EventsGeneralItem, parse_date, parse_date2
from scrapy.spiders import CSVFeedSpider
from datetime import datetime

class EventsteleticketSpider(scrapy.Spider):
    name = 'eventsTeleticket'
    custom_settings = {
        'FEED_EXPORT_FIELDS': ['nombre','ubicacion','ubicacion_detalle','ubicacion_referencia','distrito','region', \
                               'fecha_inicio','fecha_inicio_alt','fecha_fin','descripcion','precio','organizador','asistiran','me_interesa','veces_compartido','imagen','enlace_evento'],
        'DUPEFILTER_CLASS' : 'scrapy.dupefilters.BaseDupeFilter',
        'CONCURRENT_REQUESTS' : 1
    }
    allowed_domains = ['teleticket.com.pe']
    start_urls = ['https://teleticket.com.pe/musica']

    def parse(self, response):
        #//div[contains(@class, "row")] -> los eventos de musica proximos
        EVENTOS_XPATH='//article[contains(@id, "event")]'
        #Recorremos cada uno de los eventos de la Pagina de Teleticket
        for event in response.xpath(EVENTOS_XPATH):

            #EXCEPCIONES: EJ: MUSE TIENE SU PROPIA PAGINA DE EVENTOS -> https://teleticket.com.pe/muse-2019 (por eso no aparecen sus datos)
            #https://teleticket.com.pe/vivo-x-rock-2019
            #https://teleticket.com.pe/punk-rock-2019
            #https://teleticket.com.pe/black-hole-2019
            #https://teleticket.com.pe/mastercard-juanes
            #https://teleticket.com.pe/juntos-concierto-2019
            #https://teleticket.com.pe/muse-2019
            #https://teleticket.com.pe/musicales-en-concierto
            #https://teleticket.com.pe/bon-jovi-2019
            #https://teleticket.com.pe/festival-viva-salsa


            #https://teleticket.com.pe/evento/V6630  -> no tiene fecha como TEXTO
            url = event.xpath('.//a[contains(@data-ga-tracking, "ListadoEventosCategoria")]/@href').extract_first()
            yield response.follow(url, callback=self.parse_event)

    def parse_event(self, response):
        EVENT_NAME='//h1/text()'
        EVENT_WHERE='//h2/span[1]/text()'
        DATE='//h2/span[2]/text()'
        EVENT_DESCRIPTION1='//p[contains(@style, "text-align")][1]/text()[1]'
        EVENT_DESCRIPTION2='//p[contains(@style, "text-align")][1]/text()[2]'
        EVENT_DESCRIPTION3='//div[contains(@class, "resumen")][1]//p[contains(@style, "justify")][1]//b//text()'
        
        EVENT_IMAGE='//p[contains(@align, "center")]/img/@src'

        #Obtener nombre del evento
        self.nombre = response.xpath(EVENT_NAME).extract_first()

        if not self.nombre:
            self.logger.info('No se encontro nombre del evento en %s' % response.url)

            nombreArray = response.url.split('/')[-1]
            partesNombreArray = nombreArray.split('-')

            self.nombre = ""
            #print('******************************')
            #print(nombreArray)
            #print(partesNombreArray)

            for nombre in  partesNombreArray:
                self.nombre = self.nombre + " " + nombre.strip().upper()

            yield EventsGeneralItem(
                nombre=self.nombre,
                ubicacion='',
                ubicacion_detalle='',
                ubicacion_referencia='',
                distrito = '',
                region = '',
                fecha_inicio='',
                fecha_inicio_alt='',
                fecha_fin='',
                descripcion='',
                precio = '',
                organizador = '',
                asistiran = '',
                me_interesa = '',
                veces_compartido = '',
                imagen='',
                enlace_evento=response.url
            )
        else:
            self.logger.info('Parsing event %s' % self.nombre)

            #Obtener ubicación del archivo
            ubicaionObtenida = response.xpath(EVENT_WHERE).extract_first()

            if not ubicaionObtenida:
                self.logger.info('No se encontro la ubicacion del evento %s' % self.nombre)
            else:
                ubicacionArray = ubicaionObtenida.split('-')

                if len(ubicacionArray) == 3:
                    self.ubicacion = ubicacionArray[0].strip()
                    self.distrito = ubicacionArray[1].strip()
                    self.region = ubicacionArray[2].strip()
                elif len(ubicacionArray) == 2:
                    self.ubicacion = ubicacionArray[0].strip()
                    self.distrito = ubicacionArray[1].strip()
                    self.region = ubicacionArray[1].strip()
                else:
                    self.ubicacion = ubicacionArray[0].strip()
                    self.distrito = None
                    self.region = None

            #Obtener fecha del evento
            date = response.xpath(DATE).extract_first()

            if not date:
                self.logger.info('No se encontro la fecha del evento %s' % self.nombre)
            else:
                dateArray = date.split('al')

                if len(dateArray) == 2:
                    self.fecha_inicio = dateArray[0]#.replace(' ', '')
                    self.fecha_inicio_alt = dateArray[1]#.replace(' ', '')
                else:
                    self.fecha_inicio = date
                    self.fecha_inicio_alt = None

            #Obtener descripcion del evento
            descripcionParcial1 = response.xpath(EVENT_DESCRIPTION1).extract_first()
            descripcionParcial2 = response.xpath(EVENT_DESCRIPTION2).extract_first()
            descripcionParcial3 = response.xpath(EVENT_DESCRIPTION3).extract_first()
            self.descripcion = ""

            #https://teleticket.com.pe/evento/V6993 -> nueva descripcion
            if descripcionParcial1 and len(descripcionParcial1) > 0:
                #descripcion = "{} {} ".format(descripcionParcial1, descripcionParcial2).strip()  #quitar espacios al final
                self.descripcion = descripcionParcial1.strip()
            
            if len(self.descripcion) == 0 and descripcionParcial2 and len(descripcionParcial2) > 0:
                self.descripcion = descripcionParcial2.strip()

            if len(self.descripcion) == 0 and descripcionParcial3 and len(descripcionParcial3) > 0:
                self.descripcion = descripcionParcial3.strip()
            
            if len(self.descripcion) == 0:
                self.logger.info('No se encontro la descripcion del evento %s' % self.nombre)
                self.descripcion = None

            #Obtener la imagen del evento
            imagen = response.xpath(EVENT_IMAGE).extract_first()

            if not imagen:
                self.logger.info('No se encontro la imagen del evento %s' % self.nombre)
                imagen = None

            yield EventsGeneralItem(
                nombre=self.nombre,
                ubicacion=self.ubicacion,
                ubicacion_detalle='',
                ubicacion_referencia='',
                distrito = self.distrito,
                region = self.region,
                fecha_inicio=self.fecha_inicio,
                fecha_inicio_alt=self.fecha_inicio_alt,
                fecha_fin='',
                descripcion=self.descripcion,
                precio = '',
                organizador = '',
                asistiran = '',
                me_interesa = '',
                veces_compartido = '',
                imagen=imagen,
                enlace_evento=response.url
            )