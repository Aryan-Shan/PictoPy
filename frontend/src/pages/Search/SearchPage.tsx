
import { useEffect } from 'react';
import { useDispatch, useSelector } from 'react-redux';
import { useSearchParams, useNavigate } from 'react-router';
import { ImageCard } from '@/components/Media/ImageCard';
import { MediaView } from '@/components/Media/MediaView';
import { Image } from '@/types/Media';
import { setCurrentViewIndex, setImages } from '@/features/imageSlice';
import { showLoader, hideLoader } from '@/features/loaderSlice';
import { selectImages, selectIsImageViewOpen } from '@/features/imageSelectors';
import { usePictoQuery } from '@/hooks/useQueryExtension';
import { searchImages } from '@/api/api-functions/search';
import { Button } from '@/components/ui/button';
import { ArrowLeft } from 'lucide-react';
import { ROUTES } from '@/constants/routes';

export const SearchPage = () => {
    const dispatch = useDispatch();
    const navigate = useNavigate();
    const [searchParams] = useSearchParams();
    const query = searchParams.get('q');

    const isImageViewOpen = useSelector(selectIsImageViewOpen);
    const images = useSelector(selectImages);

    const { data, isLoading, isSuccess, isError, refetch } = usePictoQuery({
        queryKey: ['search-images', query],
        queryFn: async () => searchImages(query || ''),
        enabled: !!query,
    });

    useEffect(() => {
        if (!!query) {
            if (isLoading) {
                dispatch(showLoader('Searching images...'));
            } else if (isError) {
                dispatch(hideLoader());
            } else if (isSuccess) {
                const res: any = data;
                const images = (res || []) as Image[];
                dispatch(setImages(images));
                dispatch(hideLoader());
            }
        } else {
            // Clear images if no query
            dispatch(setImages([]));
        }
    }, [data, isSuccess, isError, isLoading, dispatch, query]);

    return (
        <div className="p-6 h-full overflow-y-auto">
            <div className="mb-6 flex items-center gap-4">
                <Button
                    variant="outline"
                    onClick={() => navigate(`/${ROUTES.HOME}`)}
                    className="flex cursor-pointer items-center gap-2"
                >
                    <ArrowLeft className="h-4 w-4" />
                    Back Home
                </Button>
                <h1 className="text-2xl font-bold">Search Results for "{query}"</h1>
                <span className='text-muted-foreground'> Found {images.length} results</span>
            </div>

            {(!query) && (
                <div className="flex h-[50vh] w-full items-center justify-center text-muted-foreground">
                    Please enter a search query.
                </div>
            )}

            {query && images.length === 0 && !isLoading && (
                <div className="flex h-[50vh] w-full items-center justify-center text-muted-foreground">
                    No images found.
                </div>
            )}

            <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 md:grid-cols-3 lg:grid-cols-4 xl:grid-cols-5 pb-20">
                {images.map((image, index) => (
                    <ImageCard
                        key={image.id}
                        image={image}
                        imageIndex={index}
                        className="w-full"
                        onClick={() => dispatch(setCurrentViewIndex(index))}
                    />
                ))}
            </div>

            {isImageViewOpen && <MediaView images={images} />}
        </div>
    );
};
