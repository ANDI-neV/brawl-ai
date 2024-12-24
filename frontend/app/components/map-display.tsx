"use client"
import React from 'react';
import Image from "next/image";
import { useBrawler } from './brawler-context';

const MapDisplay = () => {
    const { maps, selectedMap } = useBrawler();
    const [dimensions, setDimensions] = React.useState({ width: 0, height: 0 });

    if (!maps || !selectedMap) {
        return <div className="relative flex w-full h-[350px] md:h-[500px] bg-gray-200 rounded-2xl shadow-lg items-center justify-center font-bold text-xl">Select Map</div>;
    }

    const mapData = maps.maps[selectedMap];
    const image_url = mapData?.img_url;

    if (!mapData) {
        console.error(`Map data not found for: ${selectedMap}`);
        return <div className="relative w-full h-[350px] md:h-[500px] bg-gray-200 rounded-2xl shadow-lg">Selected map not found</div>;
    }

    const handleImageLoad = (e: React.SyntheticEvent<HTMLImageElement>) => {
        const img = e.target as HTMLImageElement;
        const containerHeight = window.innerWidth >= 768 ? 500 : 400; // match your container heights
        const containerWidth = img.parentElement?.offsetWidth || 0;
        
        // Calculate dimensions while maintaining aspect ratio
        const imageAspectRatio = img.naturalWidth / img.naturalHeight;
        let width, height;
        
        if (containerHeight * imageAspectRatio <= containerWidth) {
            // Height is the limiting factor
            height = containerHeight;
            width = containerHeight * imageAspectRatio;
        } else {
            // Width is the limiting factor
            width = containerWidth;
            height = containerWidth / imageAspectRatio;
        }

        setDimensions({ width, height });
    };

    return (
        <div className="relative h-[400px] md:h-[500px] flex items-center justify-center">
            <div 
                style={{
                    width: dimensions.width > 0 ? dimensions.width + 16 : '100%', // add 32px for padding
                    height: dimensions.height > 0 ? dimensions.height + 16 : '100%',
                }}
                className="relative bg-gray-200 rounded-2xl shadow-lg"
            >
                <div className="relative w-full h-full">
                    <Image
                        className="rounded-xl"
                        src={image_url}
                        alt={`${selectedMap} - ${mapData.game_mode}`}
                        fill
                        priority
                        sizes="(max-width: 768px) 100vw, 50vw"
                        onLoad={handleImageLoad}
                        style={{
                            objectFit: 'contain',
                            objectPosition: 'center',
                        }}
                    />
                </div>
            </div>
        </div>
    );
};

export default MapDisplay;