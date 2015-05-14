{-|
Module      : Translation.Library
Description : Put together a CUDA library file
Maintainer  : Josh Acay <cacay@cmu.edu>
Stability   : experimental
-}
module Translation.Library (compileLibrary) where

import Data.Maybe (mapMaybe)

import Text.PrettyPrint
import Text.PrettyPrint.HughesPJClass (Pretty (..), prettyShow)

import AST.AST
import AST.Types (Type (..))


data Forward = Forward Type Ident [Type]


compileLibrary :: [Fun] -> String
compileLibrary kernels = render $ headers `line` kernelFwd `line` callerFwd
  `line` library `line` lcat (map addPragma kernels) `line` pPrint callers
  where
    callers = mapMaybe caller kernels
    callerFwd = externC $ pPrint $ map forward callers
    fwd f = text (pragma f) <+> pPrint (forward f)
    kernelFwd = vcat (map fwd kernels)


headers :: Doc
headers = vcat $ map text []


forward :: Fun -> Forward
forward (Fun t id pars _) = Forward t id (map (\(Param t _) -> t) pars)


library :: Doc
library = text len
  where len = "__device__ static\n\
              \inline size_t len(void* arr)\n\
              \{\n\
              \  return *((size_t*)arr - 1);\n\
              \}"


-- | Decide whether to use __global__ or __device__ based on function properties
pragma :: Fun -> String
pragma (Fun TVoid _ _ _) = "__global__"
pragma _ = "__device__"


-- | Add __global__ or __device__ based on function properties
addPragma :: Fun -> Doc
addPragma f = text (pragma f) $+$ pPrint f


-- | Generate an invocation function for the given kernel
caller :: Fun -> Maybe Fun
caller (Fun TVoid f pars _) = Just $ Fun TVoid ("__call" ++ f) pars'
  [ Simp $ Exp $ Call (f ++ "<<<gridDim, blockDim>>>") (map Ident argNames)
  , Simp $ Exp $ Call "cudaThreadSynchronize" []
  ]
  where
    argNames = take (length pars) ["arg" ++ show i | i <- [0..]]
    pars' = Param TDim3 "gridDim" : Param TDim3 "blockDim"
      : zipWith (\(Param t _) id -> Param t id) pars argNames
caller (Fun _ _ _ _) = Nothing



-- * Pretty printing

externC :: Doc -> Doc
externC d = text "extern" <+> doubleQuotes (char 'C')
  <+> lbrace $+$ nest 2 d $+$ rbrace


line :: Doc -> Doc -> Doc
line d1 d2 = d1 <> text "\n" $+$ d2


lcat :: [Doc] -> Doc
lcat = vcat . punctuate (text "\n")


instance Pretty Forward where
  pPrint (Forward t id pars) = pPrint t <+> text id <+> pPrint pars <> semi

  pPrintList l = vcat . map (pPrintPrec l 0)


instance Show Forward where
  show = prettyShow
