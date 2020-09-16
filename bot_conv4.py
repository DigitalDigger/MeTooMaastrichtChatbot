#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 12 19:50:18 2019

@author: williamlopez
"""

#!/usr/bin/env python
# -*- coding: utf-8 -*-
# This program is dedicated to the public domain under the CC0 license.
#
# THIS EXAMPLE HAS BEEN UPDATED TO WORK WITH THE BETA VERSION 12 OF PYTHON-TELEGRAM-BOT.
# If you're still using version 11.1.0, please see the examples at
# https://github.com/python-telegram-bot/python-telegram-bot/tree/v11.1.0/examples

"""
First, a few callback functions are defined. Then, those functions are passed to
the Dispatcher and registered at their respective places.
Then, the bot is started and runs until we press Ctrl-C on the command line.
Usage:
Example of a bot-user conversation using ConversationHandler.
Send /start to initiate the conversation.
Press Ctrl-C on the command line or send a signal to the process to stop the
bot.
"""

import logging
import telegram
#from textclass import classifier
from time import sleep
from random import random
from functools import wraps
import MM_for_chatbot as mm
import pandas as pd 
from telegram import (ReplyKeyboardMarkup, ReplyKeyboardRemove, ChatAction)
from telegram.ext import (Updater, CommandHandler, MessageHandler, Filters, RegexHandler,
                          ConversationHandler)
import warnings
warnings.filterwarnings("ignore")

# Enable logging
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    level=logging.INFO)

logger = logging.getLogger(__name__)

GENDER, EVENT, LOCATION, DATE, TIME, CONFLOCATION, CONFDATE, CONFTIME, MEDICAL, ADVICE, POLICE, FINAL, FINAL2  = range(13)




def facts_to_str(user_data):
    facts = list()

    for key, value in user_data.items():
        facts.append('{} - {}'.format(key, value))

    return "\n".join(facts).join(['\n', '\n'])


def send_typing_action(func):
    """Sends typing action while processing func command."""

    @wraps(func)
    def command_func(update, context, *args, **kwargs):
        context.bot.send_chat_action(chat_id=update.effective_message.chat_id, action=ChatAction.TYPING)
        return func(update, context,  *args, **kwargs)

    return command_func




@send_typing_action
def start(update, context):
    
    evento['text'] = ' '
    evento['Target'] = -9

    Harassment = ''
    Date = ''
    Time = ''
    Location = ''
    Finaltext=''
    
    user = update.message.from_user
    #reply_keyboard = [['Boy', 'Girl', 'AI']]
    update.message.reply_text(
    'Hello, I am the #metooMaastricht bot'
    '\nSend /cancel to stop talking to me at any time, or send /start if you want to start the conversation again.\n\n')
    
    update.message.reply_text('I will ask you about your sexual assault and/or harassment experience,'
                              'I understand that this is personal. I exist to provide support and want to assure you that, I will keep all information confidential and encrypt the dialogue in this chat end to end. '
                              'Please describe your experience. The more information (including approximate time) and description that you are able to provide will allow me to direct you to the resources that can best help you.')

    return EVENT
    
    



@send_typing_action
def conflocation(update, context):
    #update.message.reply_text('confirm location')
    
    if update.message.text == 'yes':
        #Location = results.iloc[0]['Location']
        context.user_data['Location'] = Location
        update.message.reply_text('thank you for the information',
                              reply_markup=ReplyKeyboardRemove())
        if context.user_data['Date']=='':
            update.message.reply_text('on what date did it happen?',
                              reply_markup=ReplyKeyboardRemove())
            return DATE
        else:
            reply_keyboard2 = [['yes', 'no']]
            update.message.reply_text('so it happened ... ')
            update.message.reply_text(context.user_data['Date'])
            update.message.reply_text('right  ?', reply_markup=ReplyKeyboardMarkup(reply_keyboard2, one_time_keyboard=True))
            
            return CONFDATE
            
    

    else:
        update.message.reply_text('then where was it',
                              reply_markup=ReplyKeyboardRemove())
        return LOCATION
    


@send_typing_action
def confdate(update, context):
    #update.message.reply_text('confirm date')
    if update.message.text == 'yes':
        #Date = results.iloc[0]['Date']
        context.user_data['Date'] = Date
        update.message.reply_text('thank you for the information',
                              reply_markup=ReplyKeyboardRemove())
        if context.user_data['Time'] == '':
            update.message.reply_text('at what time of the day did it happen?',
                              reply_markup=ReplyKeyboardRemove())
            return TIME
        else:
            reply_keyboard2 = [['yes', 'no']]
            update.message.reply_text('so it happened at ... ')
            update.message.reply_text(context.user_data['time'])
            update.message.reply_text('right  ?', reply_markup=ReplyKeyboardMarkup(reply_keyboard2, one_time_keyboard=True))
            
            return CONFTIME
            
    

    else:
        update.message.reply_text('then when happened ?',
                              reply_markup=ReplyKeyboardRemove())
        return DATE
    

@send_typing_action
def conftime(update, context):
    #update.message.reply_text('confirm time')
    if update.message.text == 'yes':
        #Time = results.iloc[0]['Time']
        #context.user_data['Time'] = Time
        update.message.reply_text('thank you for the information',
                              reply_markup=ReplyKeyboardRemove())
        
        if context.user_data['Physical'] > 0:
            reply_keyboard2 = [['yes', 'no']]
            update.message.reply_text('Seems like you have suffered some sort of Phisical abuse')
            
            update.message.reply_text('Do you need medical assistance?', reply_markup=ReplyKeyboardMarkup(reply_keyboard2, one_time_keyboard=True))
            return MEDICAL
        
        
        elif context.user_data['Verbal'] > 0:
            update.message.reply_text('Seems like you have suffered some sort of Verbal abuse')
            update.message.reply_text('You can chat online about your experience with fier.nl (dutch only) at https://www.fier.nl/chat Monday to Friday from 7 pm to 6 am Saturday and Sunday from 8 pm to 6 am Holidays from 8 pm to 6 am, or call them 24/7 at 088 - 20 80 000 ')
         
            
            reply_keyboard2 = [['yes', 'no']]
            update.message.reply_text('Have you already reported this to the police?', reply_markup=ReplyKeyboardMarkup(reply_keyboard2, one_time_keyboard=True))
            
            return POLICE
        
        
        
        
        elif context.user_data['NonVerbal'] > 0:
            update.message.reply_text('Seems like you have suffered some sort of Non Verbal abuse')
            update.message.reply_text('you could contact "Against her will" and talk anonymously their phone is: 0592 - 34 74 44, or visit them  Monday through Thursday from 2:00 pm to 5:00 pm and from 6:00 pm to ')
            update.message.reply_text('Phone: 0592 - 34 74 44')
            update.message.reply_text('Hours: Monday through Thursday from 2:00 pm to 5:00 pm and from 6:00 pm to 9:00 pm')
            
            
            
            reply_keyboard2 = [['yes', 'no']]
            update.message.reply_text('Have you already reported this to the police?', reply_markup=ReplyKeyboardMarkup(reply_keyboard2, one_time_keyboard=True))
            
            return POLICE
       
            
    else:
        update.message.reply_text('then at what time did it happen ?',
                              reply_markup=ReplyKeyboardRemove())
        return TIME
    





def medical(update, context):
    
    if update.message.text == 'yes':
        update.message.reply_text('You can go to the Emergency Department of Maastricht UMC+, they can be contacted 7 days per week, 24 hours per day at Phone Number: 0031-43-387 67 00',
                              reply_markup=ReplyKeyboardRemove())
        
    
    update.message.reply_text('Since you have suffered phisical abuse, you can contact Centrum Seksueel Geweld Limburg (CSG Limburg) 24/7 their phone number is: 0800 01 88')
    update.message.reply_text('Or you can contact Acute care: 043 604 55 77 (for crises or emergencies), their phone is: 088 119 18 88 and they are located at: Randwycksingel 35 6229 EG Maastricht')      
    update.message.reply_text('If you are under 25 please contact: GGD Zuid Limburg-Centrum voor Seksuele Gezondheid (Burgers), their phone is: 088 880 50 72 or visit them Monday-Friday, 8:00-12:15 and Monday-Wednesday 13:30-15:30 as well',reply_markup=ReplyKeyboardRemove())


    reply_keyboard2 = [['yes', 'no']]
    
    update.message.reply_text('Have you already reported this to the police?', reply_markup=ReplyKeyboardMarkup(reply_keyboard2, one_time_keyboard=True))
            
    return POLICE
    


def police(update, context):
    if update.message.text == 'yes':
        
        update.message.reply_text('Great', reply_markup=ReplyKeyboardRemove())
        reply_keyboard2 = [['yes', 'no']]
        update.message.reply_text('Did you find this bot useful ?', reply_markup=ReplyKeyboardMarkup(reply_keyboard2, one_time_keyboard=True))
        return FINAL
    else:
        update.message.reply_text('Please report this event to the Police at this phone number: 0900 88 44')
        update.message.reply_text('And have this in mind when you report this event: https://www.politie.nl/themas/seksueel-misbruik.html ', reply_markup=ReplyKeyboardRemove())
        reply_keyboard2 = [['yes', 'no']]
        update.message.reply_text('Did you find this bot useful ?', reply_markup=ReplyKeyboardMarkup(reply_keyboard2, one_time_keyboard=True))
        return FINAL
        


def final(update, context):
    
    if update.message.text == 'yes':
        update.message.reply_text('xxxxxx')
        update.message.reply_text('good', reply_markup=ReplyKeyboardRemove())
        
        update.message.reply_text("This is the info I gathered from you:"
                              "{}"
                              "Until next time!".format(facts_to_str(context.user_data)))
        
        
        reply_keyboard2 = [['yes', 'no']]
        update.message.reply_text('To improve policy for sexual harassment and assault prevention in Maastricht and for research purposes, may we anonymously store the information you have reported? If you decline, this information will not be stored.', reply_markup=ReplyKeyboardMarkup(reply_keyboard2, one_time_keyboard=True))
        return FINAL2
    else:
        update.message.reply_text('ok, thank you for the feedback')
        reply_keyboard2 = [['yes', 'no']]
        update.message.reply_text('To improve policy for sexual harassment and assault prevention in Maastricht and for research purposes, may we anonymously store the information you have reported? If you decline, this information will not be stored.', reply_markup=ReplyKeyboardMarkup(reply_keyboard2, one_time_keyboard=True))
        return FINAL2
        
        
    

def final2(update, context):
    user_data = context.user_data
    if update.message.text == 'yes':
        update.message.reply_text('Thank you. Goodbye!', reply_markup=ReplyKeyboardRemove())
        user_data.clear()
        return ConversationHandler.END
    else:
        update.message.reply_text('Thank you. Goodbye!', reply_markup=ReplyKeyboardRemove())
        user_data.clear()
        return ConversationHandler.END
        
        


@send_typing_action
def gender(update, context):
    
    user = update.message.from_user
    logger.info("Gender of %s: %s", user.first_name, update.message.text)
    update.message.reply_text('Please tell me what happened, '
                              'so I know how to help you')

    return EVENT

@send_typing_action
def event(update, context):
    user = update.message.from_user
    user_data = context.user_data
    
    
    #photo_file = update.message.photo[-1].get_file()
    #photo_file.download('user_photo.jpg')
    
    
    
    evento['text'] = evento['text'] + ' ' + update.message.text
    #update.message.reply_text(evento['text'])
    
  
    
    logger.info("Event of %s: %s", user.first_name, evento['text'])
    
    results = mm.finale(evento['text'],10, 0.6);
    Finaltext = results.iloc[0]['text']
    Location = results.iloc[0]['Location']
    Date = results.iloc[0]['Date']
    Time = results.iloc[0]['Time']
    
    context.user_data['text'] = Finaltext
    context.user_data['Location'] = Location
    context.user_data['Date'] = Date
    context.user_data['Time'] = Time
    context.user_data['Physical'] = results.iloc[0]['Physical_flg']
    context.user_data['Verbal'] = results.iloc[0]['Verbal_flg']
    context.user_data['NonVerbal'] = results.iloc[0]['NonVerbal_flg']

    
    #update.message.reply_text(label)
    if results.iloc[0]['Harassment_flg'] == 2:
        Harassment = 'yes'
        update.message.reply_text('Seems like you have been harrassed \n ')
        
        if results.iloc[0]['Location'] == '':
            update.message.reply_text('Please indicate where this experience took place. This does not need to be precise. \n ')
            #bot.sendLocation(update.effective_message.chat_id, latitude=50.849205, longitude=5.688919, live_period='10000');
            
            return LOCATION
        else:
            #update.message.reply_text('got location')
            #user = update.message.from_user
            reply_keyboard2 = [['yes', 'no']]
            update.message.reply_text('so it happened in ')
            update.message.reply_text(results.iloc[0]['Location'])
            update.message.reply_text('right  ?', reply_markup=ReplyKeyboardMarkup(reply_keyboard2, one_time_keyboard=True))
            
            return CONFLOCATION
            
    
    elif results.iloc[0]['Harassment_flg'] == -9: 
        update.message.reply_text('I understand, can you give me more information please? ')
        return EVENT
    
    elif results.iloc[0]['Harassment_flg'] == 0: 
        update.message.reply_text('Seems like you have not been harrassed '
                              'but watch out creepers gonna creep')
        return ConversationHandler.END
    
    
    
    

@send_typing_action
def skip_event(update, context):
    user = update.message.from_user
    logger.info("User %s did not say anything.", user.first_name)
    update.message.reply_text('come on i need the info, '
                              'or send /skip.')

    return EVENT




@send_typing_action
def location(update, context):
    user_data = context.user_data
    user = update.message.from_user
    #user_location = update.message.location
    logger.info("Location of %s: %s", user.first_name, update.message.text)
    
    evento['text'] = evento['text'] + ' ' + update.message.text
    results = mm.finale(evento['text'], 10, 0.7);
    Date = results.iloc[0]['Date']
    Time = results.iloc[0]['Time']
    Location = results.iloc[0]['Location']
    Finaltext = results.iloc[0]['text']
    context.user_data['text'] = Finaltext
    context.user_data['Location'] = Location
    context.user_data['Date'] = Date
    context.user_data['Time'] = Time
    
    if results.iloc[0]['Location'] == '':
        update.message.reply_text('Can you be more explicit please?')
        return LOCATION
    else:
        reply_keyboard2 = [['yes', 'no']]
        update.message.reply_text('so it happened in ')
        update.message.reply_text(results.iloc[0]['Location'])
        update.message.reply_text('right  ?', reply_markup=ReplyKeyboardMarkup(reply_keyboard2, one_time_keyboard=True))
        return CONFLOCATION
        

@send_typing_action
def date(update, context):
    user_data = context.user_data
    user = update.message.from_user
    #user_location = update.message.location
    logger.info("Location of %s: %s", user.first_name, update.message.text)
    
    evento['text'] = evento['text'] + ' ' + update.message.text
    results = mm.finale(evento['text'],10, 0.7);
    Date = results.iloc[0]['Date']
    Time = results.iloc[0]['Time']
    Location = results.iloc[0]['Location']
    Finaltext = results.iloc[0]['text']
    context.user_data['text'] = Finaltext
    context.user_data['Location'] = Location
    context.user_data['Date'] = Date
    context.user_data['Time'] = Time
    if results.iloc[0]['Date'] == '':
        update.message.reply_text('Can you be more explicit please?')
        return DATE
    else:
        reply_keyboard2 = [['yes', 'no']]
        update.message.reply_text('so it happened in ')
        update.message.reply_text(results.iloc[0]['Date'])
        update.message.reply_text('right  ?', reply_markup=ReplyKeyboardMarkup(reply_keyboard2, one_time_keyboard=True))
        return CONFDATE
    
    
@send_typing_action
def skip_location(update, context):
    user = update.message.from_user
    logger.info("User %s did not send a location.", user.first_name)
    update.message.reply_text('ok if you dont know where, '
                              'tell me when it happened')

    return TIME  
    
    

@send_typing_action
def skip_date(update, context):
    user = update.message.from_user
    logger.info("User %s did not send a location.", user.first_name)
    update.message.reply_text('ok if you dont know where, '
                              'tell me when it happened')

    return TIME

@send_typing_action
def time(update, context):
    user_data = context.user_data
    user = update.message.from_user
    #user_location = update.message.location
    logger.info("Location of %s: %s", user.first_name, update.message.text)
    
    evento['text'] = evento['text'] + ' ' + update.message.text
    results = mm.finale(evento['text'], 10, 0.7);
    Date = results.iloc[0]['Date']
    Time = results.iloc[0]['Time']
    Location = results.iloc[0]['Location']
    Finaltext = results.iloc[0]['text']
    context.user_data['text'] = Finaltext
    context.user_data['Location'] = Location
    context.user_data['Date'] = Date
    context.user_data['Time'] = Time
    if results.iloc[0]['Time'] == '':
        update.message.reply_text('Can you be more explicit please?')
        return TIME
    else:
        reply_keyboard2 = [['yes', 'no']]
        update.message.reply_text('so it happened at ')
        update.message.reply_text(results.iloc[0]['Time'])
        update.message.reply_text('right  ?', reply_markup=ReplyKeyboardMarkup(reply_keyboard2, one_time_keyboard=True))
        return CONFTIME

@send_typing_action
def skip_time(update, context):
    user = update.message.from_user
    logger.info("User %s did not send time", user.first_name)
    update.message.reply_text('mmmmm '
                              'ok')
    return ADVICE

@send_typing_action
def advice(update, context):
    user_data = context.user_data
    user = update.message.from_user
    logger.info("advice %s: %s", user.first_name, update.message.text)
    update.message.reply_text('you need jesus')
    
    update.message.reply_text("This is the info I gathered:"
                              "{}"
                              "Until next time!".format(facts_to_str(user_data)))

    user_data.clear()
    return ConversationHandler.END
    
    



@send_typing_action
def cancel(update, context):
    user = update.message.from_user
    logger.info("User %s canceled the conversation.", user.first_name)
    update.message.reply_text('Bye! I hope we can talk again some day.',
                              reply_markup=ReplyKeyboardRemove())

    return ConversationHandler.END

@send_typing_action
def error(update, context):
    """Log Errors caused by Updates."""
    logger.warning('Update "%s" caused error "%s"', update, context.error)


evento = pd.DataFrame(index=range(1))
evento['text'] = ' '
evento['Target'] = -9
results = pd.DataFrame(index=range(1))
Harassment = ''
Date = ''
Time = ''
Location = ''
Finaltext=''
bot = telegram.Bot("755280190:AAEm9PPQPR4rhTBh2rNh2t32QgeRQO1Nusg")

def main():
    # Create the Updater and pass it your bot's token.
    # Make sure to set use_context=True to use the new context based callbacks
    # Post version 12 this will no longer be necessary
    bot = telegram.Bot("755280190:AAEm9PPQPR4rhTBh2rNh2t32QgeRQO1Nusg")
    
   
    
    
    
    updater = Updater("755280190:AAEm9PPQPR4rhTBh2rNh2t32QgeRQO1Nusg", use_context=True)
    
    

    evento = pd.DataFrame(index=range(1))
    evento['text'] = ' '
    evento['Target'] = -9

    # Get the dispatcher to register handlers
    dp = updater.dispatcher

    # Add conversation handler with the states GENDER, PHOTO, LOCATION and BIO
    conv_handler = ConversationHandler(
        entry_points=[CommandHandler('start', start)],

        states={
            GENDER: [MessageHandler(Filters.regex('^(Boy|Girl|AI)$'), gender, pass_user_data=True)],
            

            EVENT: [MessageHandler(Filters.text, event, pass_user_data=True),
                    CommandHandler('skip', skip_event)],

            LOCATION: [MessageHandler(Filters.text, location, pass_user_data=True),
                       CommandHandler('skip', skip_location)],
            DATE: [MessageHandler(Filters.text, date, pass_user_data=True),
                       CommandHandler('skip', skip_date)],

            TIME: [MessageHandler(Filters.text, time,pass_user_data=True),
                   CommandHandler('skip', skip_time)],
            
            ADVICE:[MessageHandler(Filters.text, advice)],
            
            CONFLOCATION:[MessageHandler(Filters.text, conflocation, pass_user_data=True)],
            CONFDATE:[MessageHandler(Filters.text, confdate, pass_user_data=True)],
            CONFTIME:[MessageHandler(Filters.text, conftime, pass_user_data=True)],
            POLICE:[MessageHandler(Filters.text, police, pass_user_data=True)],
            FINAL:[MessageHandler(Filters.text, final, pass_user_data=True)],
            FINAL2:[MessageHandler(Filters.text, final2, pass_user_data=True)],
            MEDICAL:[MessageHandler(Filters.text, medical, pass_user_data=True)]
            
        },

        fallbacks=[CommandHandler('cancel', cancel)]
    )

    dp.add_handler(conv_handler)

    # log all errors
    dp.add_error_handler(error)

    # Start the Bot
    updater.start_polling()

    # Run the bot until you press Ctrl-C or the process receives SIGINT,
    # SIGTERM or SIGABRT. This should be used most of the time, since
    # start_polling() is non-blocking and will stop the bot gracefully.
    updater.idle()


if __name__ == '__main__':
    main()