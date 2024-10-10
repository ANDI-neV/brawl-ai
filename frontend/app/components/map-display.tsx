"use client"
import React from 'react';
import Image from "next/image";
import { useBrawler } from './brawler-context';
import { MapInterface } from './api-handler';

const MapDisplay = () => {
    const { maps, selectedMap } = useBrawler();

    if (!maps || !selectedMap) {
        return <div className="relative flex w-full h-[500px] bg-gray-200 rounded-2xl shadow-lg items-center justify-center font-bold text-xl">Select Map</div>;
    }

    const mapData = maps.maps[selectedMap];
    const image_url = mapData?.img_url;


    if (!mapData) {
        console.error(`Map data not found for: ${selectedMap}`);
        return <div className="relative w-full h-[500px] bg-gray-200 rounded-2xl shadow-lg">Selected map not found</div>;
    }
    return (
        <div className="relative w-full h-[500px] bg-gray-200 rounded-2xl shadow-lg">
            <Image
                className="rounded-xl object-cover object-left"
                src={image_url}
                alt={`${selectedMap} - ${mapData.game_mode}`}
                fill
                sizes="(max-width: 768px) 100vw, 50vw"
                style={{
                    objectFit: 'cover',
                    objectPosition: 'center'
                }}
            />
        </div>
    )
}

export default MapDisplay;