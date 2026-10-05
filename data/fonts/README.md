# Fonts

The model was trained on 1,073 Persian fonts. Only the 49 of them that are published under a free
licence are in this repository. The others come from a collection whose terms of redistribution could not be
verified, so they are not distributed here. `../fonts_manifest.csv` lists all 1,073 with their file name and
SHA-256 checksum, and `../fonts.csv` is the annotation file the training reads. Fonts whose files are missing
are skipped with a message, so training runs on whatever part of the collection is present.

The font index in `pretrained/fonts.sqlite` covers all 1,073 fonts. It holds latent vectors and names, no
font data.

| file | font | licence | copyright notice in the file |
|------|------|---------|------------------------------|
| `ReemKufi.ttf` | Reem Kufi Regular | SIL Open Font License 1.1 | Copyright 2015-2022 The Reem Kufi Project Authors (https://github.com/aliftype/reem-kufi). |
| `Vazirmatn[wght].ttf` | Vazirmatn Regular | SIL Open Font License 1.1 | Copyright 2015 The Vazirmatn Project Authors (https://github.com/rastikerdar/vazirmatn) |
| `ArefRuqaa-Regular.ttf` | Aref Ruqaa Regular | SIL Open Font License 1.1 | Copyright 2015-2022 The Aref Ruqaa Project Authors (https://github.com/aliftype/aref-ruqaa), with Reserved Font Name EURM10. |
| `Amiri-Regular.ttf` | Amiri Regular | SIL Open Font License 1.1 | Copyright 2010-2022 The Amiri Project Authors (https://github.com/aliftype/amiri). |
| `SahelFD-Light.ttf` | Sahel FD Light | SIL Open Font License 1.1 | Copyright (c) 2016 by Saber Rastikerdar. All Rights Reserved. |
| `Vazirmatn-RD-Thin.ttf` | Vazirmatn RD Thin | SIL Open Font License 1.1 | Copyright 2015 The Vazirmatn Project Authors (https://github.com/rastikerdar/vazirmatn) |
| `Vazirmatn-RD-ExtraBold.ttf` | Vazirmatn RD ExtraBold | SIL Open Font License 1.1 | Copyright 2015 The Vazirmatn Project Authors (https://github.com/rastikerdar/vazirmatn) |
| `SahelVF-Regular.ttf` | Sahel VF Regular | SIL Open Font License 1.1 | Copyright (c) 2016 by Saber Rastikerdar. All Rights Reserved. |
| `Amiri-Bold.ttf` | Amiri Bold | SIL Open Font License 1.1 | Copyright 2010-2022 The Amiri Project Authors (https://github.com/aliftype/amiri). |
| `ParastooFD-Bold.ttf` | Parastoo FD Bold | SIL Open Font License 1.1 | Copyright (c) 2016 by Saber Rastikerdar. All Rights Reserved. |
| `SamimFD-Medium.ttf` | Samim FD Medium | SIL Open Font License 1.1 | Copyright (c) 2003 by Bitstream, Inc. All Rights Reserved. |
| `Vazirmatn-RD-Medium.ttf` | Vazirmatn RD Medium | SIL Open Font License 1.1 | Copyright 2015 The Vazirmatn Project Authors (https://github.com/rastikerdar/vazirmatn) |
| `Amiri-Italic.ttf` | Amiri Italic | SIL Open Font License 1.1 | Copyright 2010-2022 The Amiri Project Authors (https://github.com/aliftype/amiri). |
| `Vazirmatn-Medium.ttf` | Vazirmatn Medium | SIL Open Font License 1.1 | Copyright 2015 The Vazirmatn Project Authors (https://github.com/rastikerdar/vazirmatn) |
| `Vazirmatn-Light.ttf` | Vazirmatn Light | SIL Open Font License 1.1 | Copyright 2015 The Vazirmatn Project Authors (https://github.com/rastikerdar/vazirmatn) |
| `Vazirmatn-SemiBold.ttf` | Vazirmatn SemiBold | SIL Open Font License 1.1 | Copyright 2015 The Vazirmatn Project Authors (https://github.com/rastikerdar/vazirmatn) |
| `Vazirmatn-RD-Regular.ttf` | Vazirmatn RD Regular | SIL Open Font License 1.1 | Copyright 2015 The Vazirmatn Project Authors (https://github.com/rastikerdar/vazirmatn) |
| `GandomFD-Regular.ttf` | Gandom FD Regular | SIL Open Font License 1.1 | Copyright (c) 2003 by Bitstream, Inc. All Rights Reserved. |
| `Vazirmatn-RD[wght].ttf` | Vazirmatn RD Regular | SIL Open Font License 1.1 | Copyright 2015 The Vazirmatn Project Authors (https://github.com/rastikerdar/vazirmatn) |
| `Vazirmatn-ExtraLight.ttf` | Vazirmatn ExtraLight | SIL Open Font License 1.1 | Copyright 2015 The Vazirmatn Project Authors (https://github.com/rastikerdar/vazirmatn) |
| `NahidFD-Regular.ttf` | Nahid FD Regular | Bitstream Vera Fonts licence, changes in the public domain | Copyright (c) 2003 by Bitstream, Inc. All Rights Reserved. |
| `SahelFD-SemiBold.ttf` | Sahel FD SemiBold | SIL Open Font License 1.1 | Copyright (c) 2016 by Saber Rastikerdar. All Rights Reserved. |
| `ShabnamFD-Regular.ttf` | Shabnam FD Regular | SIL Open Font License 1.1 | Copyright (c) 2003 by Bitstream, Inc. All Rights Reserved. |
| `Vazirmatn-Thin.ttf` | Vazirmatn Thin | SIL Open Font License 1.1 | Copyright 2015 The Vazirmatn Project Authors (https://github.com/rastikerdar/vazirmatn) |
| `Amiri-BoldItalic.ttf` | Amiri Bold Italic | SIL Open Font License 1.1 | Copyright 2010-2022 The Amiri Project Authors (https://github.com/aliftype/amiri). |
| `SamimFD-Bold.ttf` | Samim FD Bold | SIL Open Font License 1.1 | Copyright (c) 2003 by Bitstream, Inc. All Rights Reserved. |
| `ArefRuqaa-Bold.ttf` | Aref Ruqaa Bold | SIL Open Font License 1.1 | Copyright 2015-2022 The Aref Ruqaa Project Authors (https://github.com/aliftype/aref-ruqaa), with Reserved Font Name EURM10. |
| `ShabnamFD-Medium.ttf` | Shabnam FD Medium | SIL Open Font License 1.1 | Copyright (c) 2003 by Bitstream, Inc. All Rights Reserved. |
| `Vazirmatn-Black.ttf` | Vazirmatn Black | SIL Open Font License 1.1 | Copyright 2015 The Vazirmatn Project Authors (https://github.com/rastikerdar/vazirmatn) |
| `SahelFD-Bold.ttf` | Sahel FD Bold | SIL Open Font License 1.1 | Copyright (c) 2016 by Saber Rastikerdar. All Rights Reserved. |
| `Vazirmatn-RD-Bold.ttf` | Vazirmatn RD Bold | SIL Open Font License 1.1 | Copyright 2015 The Vazirmatn Project Authors (https://github.com/rastikerdar/vazirmatn) |
| `ShabnamFD-Light.ttf` | Shabnam FD Light | SIL Open Font License 1.1 | Copyright (c) 2003 by Bitstream, Inc. All Rights Reserved. |
| `Vazirmatn-Regular.ttf` | Vazirmatn Regular | SIL Open Font License 1.1 | Copyright 2015 The Vazirmatn Project Authors (https://github.com/rastikerdar/vazirmatn) |
| `Vazirmatn-Bold.ttf` | Vazirmatn Bold | SIL Open Font License 1.1 | Copyright 2015 The Vazirmatn Project Authors (https://github.com/rastikerdar/vazirmatn) |
| `ParastooFD-Regular.ttf` | Parastoo FD Regular | SIL Open Font License 1.1 | Copyright (c) 2016 by Saber Rastikerdar. All Rights Reserved. |
| `ShabnamFD-Bold.ttf` | Shabnam FD Bold | SIL Open Font License 1.1 | Copyright (c) 2003 by Bitstream, Inc. All Rights Reserved. |
| `Vazirmatn-RD-Light.ttf` | Vazirmatn RD Light | SIL Open Font License 1.1 | Copyright 2015 The Vazirmatn Project Authors (https://github.com/rastikerdar/vazirmatn) |
| `VazirCodeExtraHeightFD-Regular.ttf` | Vazir Code Extra Height FD Regular | SIL Open Font License 1.1 | Copyright (c) 2003 by Bitstream, Inc. All Rights Reserved. |
| `Vazirmatn-RD-SemiBold.ttf` | Vazirmatn RD SemiBold | SIL Open Font License 1.1 | Copyright 2015 The Vazirmatn Project Authors (https://github.com/rastikerdar/vazirmatn) |
| `Vazirmatn-RD-ExtraLight.ttf` | Vazirmatn RD ExtraLight | SIL Open Font License 1.1 | Copyright 2015 The Vazirmatn Project Authors (https://github.com/rastikerdar/vazirmatn) |
| `SamimFD-Regular.ttf` | Samim FD Regular | SIL Open Font License 1.1 | Copyright (c) 2003 by Bitstream, Inc. All Rights Reserved. |
| `SahelFD-Black.ttf` | Sahel FD Black | SIL Open Font License 1.1 | Copyright (c) 2016 by Saber Rastikerdar. All Rights Reserved. |
| `TanhaFD-Regular.ttf` | Tanha FD Regular | Bitstream Vera Fonts licence, changes in the public domain | Copyright (c) 2003 by Bitstream, Inc. All Rights Reserved. |
| `Vazirmatn-RD-Black.ttf` | Vazirmatn RD Black | SIL Open Font License 1.1 | Copyright 2015 The Vazirmatn Project Authors (https://github.com/rastikerdar/vazirmatn) |
| `VazirCodeHackExtraHeightFD-Regular.ttf` | Vazir Code Hack Extra Height FD Regular | SIL Open Font License 1.1 | Copyright (c) 2003 by Bitstream, Inc. All Rights Reserved. |
| `ShabnamFD-Thin.ttf` | Shabnam FD Thin | SIL Open Font License 1.1 | Copyright (c) 2003 by Bitstream, Inc. All Rights Reserved. |
| `SahelFD-Regular.ttf` | Sahel FD Regular | SIL Open Font License 1.1 | Copyright (c) 2016 by Saber Rastikerdar. All Rights Reserved. |
| `Vazirmatn-ExtraBold.ttf` | Vazirmatn ExtraBold | SIL Open Font License 1.1 | Copyright 2015 The Vazirmatn Project Authors (https://github.com/rastikerdar/vazirmatn) |
| `ReemKufiInk.ttf` | Reem Kufi Ink Regular | SIL Open Font License 1.1 | Copyright 2015-2022 The Reem Kufi Project Authors (https://github.com/aliftype/reem-kufi). |

The text of the SIL Open Font License is in [OFL.txt](OFL.txt). The fonts under the Bitstream Vera Fonts
licence carry its full text in their name table.
