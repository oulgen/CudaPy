{-|
Module      : Util.Error
Description : Utility functions for error handling
Maintainer  : Josh Acay <cacay@cmu.edu>
Stability   : experimental
-}
module Util.Error (liftEIO, assertMsg, liftMaybe, msum1) where

import Control.Monad.Error


liftEIO :: Either String a -> ErrorT String IO a
liftEIO (Left s)  = throwError s
liftEIO (Right x) = return x


assertMsg :: MonadError e m => e -> Bool -> m ()
assertMsg e b = if b then return () else throwError e


liftMaybe :: MonadError e m => e -> Maybe a -> m a
liftMaybe e Nothing = throwError e
liftMaybe _ (Just x) = return x


msum1 :: MonadPlus m => [m a] -> m a
msum1 = foldr1 mplus