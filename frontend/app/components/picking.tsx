"use client"
import TeamDisplay from "./team-display";
import BrawlerPicker from "./brawler-picker";
import MapDisplay from "./map-display";
import BrawlerBans from "./brawler-bans";

const Picking = () => {
    return (
        <div className="container mx-auto px-4 py-8 max-w-full">
            <div className="flex flex-col lg:flex-row lg:justify-center lg:items-start gap-8">
                <div className="w-full lg:w-auto flex flex-col justify-center items-center relative">
                    <div className="z-20">
                        <TeamDisplay team="right" bgColor="#ef4444"/>
                    </div>
                    <div className="w-full h-full z-10 -my-16">
                        <MapDisplay/>
                    </div>
                    <div className="z-20">
                        <TeamDisplay team="left" bgColor="#3b82f6"/>
                    </div>
                </div>
                <div className="my-auto"> 
                    <BrawlerBans/>
                </div>
                <div className="w-full lg:w-auto flex-grow lg:max-w-2xl xl:max-w-3xl">
                    <BrawlerPicker/>
                </div>
            </div>
        </div>
    )
}


export default Picking;