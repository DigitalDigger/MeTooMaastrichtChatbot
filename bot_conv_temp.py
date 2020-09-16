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

# Enable logging
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    level=logging.INFO)

logger = logging.getLogger(__name__)

GENDER, EVENT, LOCATION, DATE, TIME, CONFLOCATION, CONFDATE, CONFTIME, ADVICE = range(9)

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
    user = update.message.from_user
    reply_keyboard = [['Boy', 'Girl', 'AI']]
    update.message.reply_text(
    'Hello, I am the #metooMaastricht bot and would like to help you'
    '\nSend /cancel to stop talking to me.\n\n'
    'Are you a boy or a girl ?',
    reply_markup=ReplyKeyboardMarkup(reply_keyboard, one_time_keyboard=True))
    
    
    return GENDER



@send_typing_action
def conflocation(update, context):
    #update.message.reply_text('confirm location')
    
    if update.message.text == 'yes':
        #Location = results.iloc[0]['Location']
        context.user_data['Location'] = Location
        update.message.reply_text('thanks! for the info',
                              reply_markup=ReplyKeyboardRemove())
        if Date=='':
            update.message.reply_text('when did it happen?',
                              reply_markup=ReplyKeyboardRemove())
            return DATE
        else:
            reply_keyboard2 = [['yes', 'no']]
            update.message.reply_text('so it happened ... ')
            update.message.reply_text(results.iloc[0]['Date'])
            update.message.reply_text('right  ?', reply_markup=ReplyKeyboardMarkup(reply_keyboard2, one_time_keyboard=True))
            
            return CONFDATE
            
    

    else:
        update.message.reply_text('then where was it',
                              reply_markup=ReplyKeyboardRemove())
        return LOCATION
    



def confdate(update, context):
    #update.message.reply_text('confirm date')
    if update.message.text == 'yes':
        #Date = results.iloc[0]['Date']
        context.user_data['Date'] = Date
        update.message.reply_text('thanks! for the info',
                              reply_markup=ReplyKeyboardRemove())
        if Time == '':
            update.message.reply_text('at what time did it happen?',
                              reply_markup=ReplyKeyboardRemove())
            return TIME
        else:
            reply_keyboard2 = [['yes', 'no']]
            update.message.reply_text('so it happened at ... ')
            update.message.reply_text(results.iloc[0]['time'])
            update.message.reply_text('right  ?', reply_markup=ReplyKeyboardMarkup(reply_keyboard2, one_time_keyboard=True))
            
            return CONFTIME
            
    

    else:
        update.message.reply_text('then when happened ?',
                              reply_markup=ReplyKeyboardRemove())
        return DATE
    


def conftime(update, context):
    #update.message.reply_text('confirm time')
    if update.message.text == 'yes':
        #Time = results.iloc[0]['Time']
        #context.user_data['Time'] = Time
        update.message.reply_text('thanks! for the info',
                              reply_markup=ReplyKeyboardRemove())
        
        return ADVICE
       
            

    else:
        update.message.reply_text('then at what time did it happen ?',
                              reply_markup=ReplyKeyboardRemove())
        return TIME
    

@send_typing_action
def gender(update, context):
    
    user = update.message.from_user
    logger.info("Gender of %s: %s", user.first_name, update.message.text)
    update.message.reply_text('I see! Please tell me what happened, '
                              'so I know how to help you, or send /skip if you don\'t want to.',
                              reply_markup=ReplyKeyboardRemove())

    return EVENT

@send_typing_action
def event(update, context):
    user = update.message.from_user
    user_data = context.user_data
    #photo_file = update.message.photo[-1].get_file()
    #photo_file.download('user_photo.jpg')
    
    
    
    evento['text'] = evento['text'] + ' ' + update.message.text
    
  
    
    logger.info("Event of %s: %s", user.first_name, evento['text'])
    
    results = mm.finale(evento, 'text', 'processed', 'Target', 10, 0.7);
    Finaltext = results.iloc[0]['text']
    Location = results.iloc[0]['Location']
    Date = results.iloc[0]['Date']
    Time = results.iloc[0]['Time']
    
    context.user_data['text'] = Finaltext
    context.user_data['Location'] = Location
    context.user_data['Date'] = Date
    context.user_data['Time'] = Time

    
    #update.message.reply_text(label)
    if results.iloc[0]['Harassment_flg'] == 2:
        Harassment = 'yes'
        update.message.reply_text('Seems like you have been harrassed \n ')
        
        if results.iloc[0]['Location'] == '':
            update.message.reply_text('Where did it happen ? \n ')
            return LOCATION
        else:
            #update.message.reply_text('got location')
            #user = update.message.from_user
            reply_keyboard2 = [['yes', 'no']]
            update.message.reply_text('so it happened in ')
            update.message.reply_text(results.iloc[0]['Location'])
            update.message.reply_text('right  ?', reply_markup=ReplyKeyboardMarkup(reply_keyboard2, one_time_keyboard=True))
            
            return CONFLOCATION
            
    
    elif results.iloc[0]['Harassment_flg'] == 1: 
        update.message.reply_text('I see, can you give me more details '
                              'I need more info!')
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
    results = mm.finale(evento, 'text', 'processed', 'Target', 10, 0.7);
    Date = results.iloc[0]['Date']
    Time = results.iloc[0]['Time']
    Location = results.iloc[0]['Location']
    Finaltext = results.iloc[0]['text']
    context.user_data['text'] = Finaltext
    context.user_data['Location'] = Location
    context.user_data['Date'] = Date
    context.user_data['Time'] = Time
    
    if results.iloc[0]['Location'] == '':
        update.message.reply_text('I am sorry my NER system is lazy, can you be more explicit ?')
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
    results = mm.finale(evento, 'text', 'processed', 'Target', 10, 0.7);
    Date = results.iloc[0]['Date']
    Time = results.iloc[0]['Time']
    Location = results.iloc[0]['Location']
    Finaltext = results.iloc[0]['text']
    context.user_data['text'] = Finaltext
    context.user_data['Location'] = Location
    context.user_data['Date'] = Date
    context.user_data['Time'] = Time
    if results.iloc[0]['Date'] == '':
        update.message.reply_text('I am sorry my NER system is lazy, can you be more explicit ?')
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
    results = mm.finale(evento, 'text', 'processed', 'Target', 10, 0.7);
    Date = results.iloc[0]['Date']
    Time = results.iloc[0]['Time']
    Location = results.iloc[0]['Location']
    Finaltext = results.iloc[0]['text']
    context.user_data['text'] = Finaltext
    context.user_data['Location'] = Location
    context.user_data['Date'] = Date
    context.user_data['Time'] = Time
    if results.iloc[0]['Time'] == '':
        update.message.reply_text('I am sorry my NER system is lazy, can you be more explicit ?')
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

def main():
    # Create the Updater and pass it your bot's token.
    # Make sure to set use_context=True to use the new context based callbacks
    # Post version 12 this will no longer be necessary
    
    
   
    
    
    
    updater = Updater("755280190:AAEm9PPQPR4rhTBh2rNh2t32QgeRQO1Nusg", use_context=True)
    
    
    context.user_data['evento'] = pd.DataFrame(index=range(1))
    context.user_data.evento['text']= ' '
    context.user_data.evento['Target']= -9
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
            CONFTIME:[MessageHandler(Filters.text, conftime, pass_user_data=True)]
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