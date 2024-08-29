"use client"
import TeamDisplay from "./team-display";
import BrawlerPicker from "./brawler-picker";

const Picking = () => {
    return (
        <div className="container mx-auto px-4 py-8 max-w-full">
            <div className="flex flex-col lg:flex-row lg:justify-center lg:items-start gap-24">
                <div className="w-full lg:w-auto flex flex-row justify-center items-center gap-4 sm:gap-8 self-center">
                    <TeamDisplay team="left" bgColor="#3b82f6"/>
                    <TeamDisplay team="right" bgColor="#f43f5e"/>
                </div>
                <div className="w-full lg:w-auto flex-grow lg:max-w-2xl xl:max-w-3xl">
                    <BrawlerPicker/>
                </div>
            </div>
        </div>
    )
}

export default Picking;