from nicegui import ui, app
import aimanager
from db import Database
import ai


playersarray = ["a1", "b1", "b2", "a2", "a3", "b3"]
aidict = {}
brawlers = ai.prepare_brawler_data()
availablebrawlers = []
availablebrawlers = list(brawlers.keys())
db = Database()
availablemaps = db.getAllMaps()
availablemaps = [x[0] for x in availablemaps]
us = True

def set_first_selection(selector):
    global us
    if selector.value == 'Us':
        playersarray[0] = "a1"
        playersarray[1] = "b1"
        playersarray[2] = "b2"
        playersarray[3] = "a2"
        playersarray[4] = "a3"
        playersarray[5] = "b3"
    else:
        playersarray[0] = "b1"
        playersarray[1] = "a1"
        playersarray[2] = "a2"
        playersarray[3] = "b2"
        playersarray[4] = "b3"
        playersarray[5] = "a3"
    if selector.value == 'Us':
        us = True
    else:
        us = False
    add_stepper(0)


def add_brawler(num, selector):
    #playersarray[num] = selector.value
    aidict[playersarray[num]] = selector.value
    print(selector.value)
    availablebrawlers.remove(selector.value)
    brawlerslabel.text = str(playersarray)
    aidictl.text = str(aidict)

def print_winrate():
    global wrlabel
    wr = aimanager.get_finished_winrate(us, aidict, mapselector.value)
    wrlabel.text = f"Final Winrate: {wr:.3f}"

def obtain_next_pick():
    best_picks = aimanager.get_next_pick(us, aidict, mapselector.value)
    global dataseries
    global categories
    dataseries = []
    all = []
    for i in range(best_picks.__len__()):
        all.append(best_picks[i][1])
    dataseries.append({'data': all})
    categories = []
    for i in range(best_picks.__len__()):
        categories.append(best_picks[i][0])
    global chart
    chart.options['xAxis']['categories'] = categories
    chart.options['series'] = dataseries
    chart.update()
    print("Updated with" + str(categories) + " and " + str(dataseries))
    



def add_stepper(num):
    with stepper:
        with ui.step(playersarray[num]) as currentstep:
            selecter = ui.select(availablebrawlers, with_input=True)

            def callback():
                add_brawler(num, selecter)
                if num < 5:
                    add_stepper(num + 1)
                    if(us and (num == 2 or num == 3)) or (not us and (num == 0 or num == 1 or num == 4)):
                        obtain_next_pick()
                else:
                    print_winrate()
                stepper.next()

            ui.button('Next', on_click=callback)
        currentstep.move(target_index=num+2)

dataseries = []
all = []
for i in range(availablebrawlers.__len__()):
    all.append(i)

dataseries.append({'data': all})
    
   

if __name__ in {"__main__", "__mp_main__"}:
    ui.page_title('Brawl AI')

    ui.label('Do the thing')

    global stepper
    with ui.stepper().props('vertical').classes('w-full') as stepper:
        with ui.step('Select map'):
            global mapselector
            mapselector = ui.select(availablemaps, value=availablemaps[0])
            ui.button('Next', on_click=stepper.next)
        with ui.step('Which side first?'):

            def callback():
                set_first_selection(startselector)
                stepper.next()

            startselector = ui.toggle(['Us', 'Them'])
            ui.button('Next', on_click=callback)

        with ui.step('Finished'):
            ui.label('Finished')

    global brawlerslabel
    brawlerslabel = ui.label()
    global aidictl
    aidictl = ui.label()
    global wrlabel
    wrlabel = ui.label("Winrate")


    categories = availablebrawlers
    global chart
    chart = ui.highchart({
        'title': {'text': 'Brawlers'},
        'chart': {'type': 'bar', 'height': '800px'},
        'xAxis': {'categories': categories},
        'series': dataseries,
    })

    ui.button('Shutdown', on_click=app.shutdown)

    ui.run(reload=False)